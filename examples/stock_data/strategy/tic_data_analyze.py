import math
import os
import statistics
from datetime import datetime, timedelta
import traceback
from functools import lru_cache
from io import TextIOWrapper, StringIO
from pathlib import Path
from time import sleep

import akshare as ak
import numpy as np
import pandas as pd
import requests

# Tushare（中国股市专用）：提供A股、港股的逐笔成交、订单簿数据，需注册获取Token
import tushare as ts
from akshare import stock_info_sz_name_code, stock_info_bj_name_code, stock_info_sh_name_code
from matplotlib import pyplot as plt
from pandas import DataFrame
from tqdm import tqdm


class MainForceAnalyzer:
    '''
        主力行为分析类

        dataFrame的数据结构：
        成交明细的数据结构：
        TIME：成交时间
        PRICE：成交价格
        VOLUME：成交量
        TYPE：成交类型（买盘：买涨 卖盘：卖跌 中性盘：不涨不跌）
    '''

    def __init__(self, df):
        self.df = df.copy()
        self.analyze()

    @staticmethod
    def split_to_windows(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """ 按照交易记录条数拆分数据到指定窗口大小 """
        for i in range(0, len(df), window_size):
            yield i / window_size, df.iloc[i: i + window_size]

    @staticmethod
    def split_by_time_window(df, time_col, window=5):
        """按时间窗口拆分 DataFrame  默认五分钟"""
        df[time_col] = pd.to_datetime(df[time_col])  # 确保是 datetime
        # df = df.sort_values(time_col)  # 按时间排序

        # 计算分组（每 五 分钟一组）
        df["group"] = ((df[time_col].astype('int64') // 10 ** 9).astype('int64') // (np.int64(window) * 60)).astype(
            "int")
        # 按分组拆分
        chunks = [group for _, group in df.groupby("group")]
        # 删除临时分组列
        for chunk in chunks:
            chunk.drop("group", axis=1, inplace=True)
        return chunks

    def analyze(self):
        """执行完整的主力分析"""
        self.df["TIME"] = pd.to_datetime(self.df['TIME'], format="%H:%M:%S")
        self.calculate_order_size()
        self.classify_orders()
        self.identify_main_force_patterns(5)

    def calculate_order_size(self):
        """计算每笔成交的订单规模"""
        # 计算每笔成交的金额（万元） 找到成交量和成交额的分位数，获取到讲数据0.8或者0.92拆分的数值
        self.df['AMOUNT_WAN'] = self.df['AMOUNT'] / 10000
        large_volume_threshold = self.df['VOLUME'].quantile(0.8)
        huge_volume_threshold = self.df['VOLUME'].quantile(0.92)
        large_threshold = self.df['AMOUNT_WAN'].quantile(0.8)  # 找出
        huge_threshold = self.df['AMOUNT_WAN'].quantile(0.92)

        # 定义大单阈值（可根据股票流通盘调整）
        self.df['IS_LARGE_ORDER'] = self.df.query(
            f"(AMOUNT_WAN >= {large_threshold} and AMOUNT_WAN < {huge_threshold}) or (VOLUME >= {large_volume_threshold} and VOLUME < {huge_volume_threshold})")[
            "AMOUNT_WAN"]  # 50万元以上的算大单
        self.df['IS_HUGE_ORDER'] = (self.df['AMOUNT_WAN'] >= huge_threshold).astype(bool) | (
                    self.df['VOLUME'] >= huge_volume_threshold).astype(bool)  # 200万元以上的算特大单

        # print("大单统计：")
        # print(f"总成交笔数: {len(self.df)}")
        # print(f"大单笔数: {self.df['IS_LARGE_ORDER'].sum()}")
        # print(f"特大单笔数: {self.df['IS_HUGE_ORDER'].sum()}")

    def classify_orders(self):
        """分类资金流向"""
        # 根据买卖类型标记资金流向
        buy_condition = self.df['TYPE'].str.contains('买盘')
        sell_condition = self.df['TYPE'].str.contains('卖盘')
        neutral_condition = self.df['TYPE'].str.contains('中性盘')

        # 计算20笔交易的价格变化是否是正向的
        trade_num = 20
        self.df[f'PRICE_CHANGE_POSITIVE_{trade_num}_TRADE'] = self.df['PRICE'].pct_change(periods=20).fillna(0).replace(
            [np.inf, -np.inf], 0) * 100 >= 0

        # 计算买单和卖单成交的金额 计算方向
        self.df['DIRECTION'] = np.where(
            buy_condition | (neutral_condition & self.df[f'PRICE_CHANGE_POSITIVE_{trade_num}_TRADE']), 'B', 'S')
        self.df['BUY_AMOUNT'] = np.where(
            buy_condition | (neutral_condition & self.df[f'PRICE_CHANGE_POSITIVE_{trade_num}_TRADE']),
            self.df['AMOUNT_WAN'], 0)
        self.df['SELL_AMOUNT'] = np.where(
            sell_condition | (neutral_condition & ~self.df[f'PRICE_CHANGE_POSITIVE_{trade_num}_TRADE']),
            self.df['AMOUNT_WAN'], 0)

        # 大单资金流向
        self.df['LARGE_BUY'] = np.where((self.df['BUY_AMOUNT'] > 0).astype(bool) & self.df['IS_LARGE_ORDER'],
                                        self.df['AMOUNT_WAN'], 0)
        self.df['LARGE_SELL'] = np.where((self.df['SELL_AMOUNT'] > 0).astype(bool) & self.df['IS_LARGE_ORDER'],
                                         self.df['AMOUNT_WAN'], 0)

        # 特大单资金流向
        self.df['HUGE_BUY'] = np.where((self.df['BUY_AMOUNT'] > 0).astype(bool) & self.df['IS_HUGE_ORDER'],
                                       self.df['AMOUNT_WAN'], 0)
        self.df['HUGE_SELL'] = np.where((self.df['SELL_AMOUNT'] > 0).astype(bool) & self.df['IS_HUGE_ORDER'],
                                        self.df['AMOUNT_WAN'], 0)

        # 将loc函数self.df['LARGE_BUY'] | self.df['HUGE_BUY']作为条件，然后将等号后面的值赋予LARGE_NET_FLOW
        self.df.loc[
            (self.df['LARGE_BUY'] > 0).astype(bool) | (self.df['HUGE_BUY'] > 0).astype(bool), 'LARGE_NET_FLOW'] = \
        self.df['AMOUNT_WAN']
        self.df.loc[
            (self.df['LARGE_SELL'] > 0).astype(bool) | (self.df['HUGE_SELL'] > 0).astype(bool), 'LARGE_NET_FLOW'] = - \
        self.df['AMOUNT_WAN']

    def identify_main_force_patterns(self, rolling_window: int = 5):
        """简单识别主力操作模式 准备主力相关数据"""
        # 计算滚动窗口内的主力行为
        # 5笔成交为一个分析窗口
        # 主力净流入
        self.df['NET_MAIN_INFLOW'] = (self.df['LARGE_BUY'] + self.df['HUGE_BUY'] -
                                      self.df['LARGE_SELL'] - self.df['HUGE_SELL'])
        # 主力净流入滚动均值
        self.df['MAIN_INFLOW_MA'] = self.df['NET_MAIN_INFLOW'].rolling(window=rolling_window).mean()

        # 简单识别主力吸筹模式（价格平稳或下跌时大单买入）
        self.df['PRICE_CHANGE'] = self.df['PRICE'].pct_change() * 100
        self.df['ACCUMULATION_PATTERN'] = (
                (self.df['NET_MAIN_INFLOW'] > 100) &
                (self.df['PRICE_CHANGE'] < 0.5)  # 价格涨幅小于0.5%
        )
        # 简单识别主力出货模式（价格上涨时大单卖出）
        self.df['DISTRIBUTION_PATTERN'] = (
                (self.df['NET_MAIN_INFLOW'] < -100) &
                (self.df['PRICE_CHANGE'] > 0.5)  # 价格涨幅大于0.5%
        )

    def advanced_main_force_pattern_recognition(self):
        """
        高级模式识别 中单吸筹  大单撤单分析等
        """
        # 1. 隐形大单识别（多笔连续中等买单）
        self.df['MEDIUM_ORDER'] = (self.df['VOLUME'] > self.df['VOLUME'].quantile(0.6)) & \
                                  (self.df['VOLUME'] < self.df['VOLUME'].quantile(0.8))
        self.df['MEDIUM_BUY_SEQUENCE'] = (
            (self.df['MEDIUM_ORDER'] & (self.df['DIRECTION'] == 'B')).astype(int)
        )
        # 识别连续中等买单  然后给一个分组编号
        self.df['MEDIUM_BUY_GROUP'] = (self.df['MEDIUM_BUY_SEQUENCE'].diff() != 0).cumsum()
        medium_buy_groups = self.df[self.df['MEDIUM_BUY_SEQUENCE'] == 1].groupby('MEDIUM_BUY_GROUP').agg({
            'VOLUME': ['count', 'sum'],
            'AMOUNT_WAN': 'sum',
            'PRICE': ['min', 'max', 'last']
        })
        medium_buy_groups.columns = ['VOLUME_COUNT', 'TOTAL_VOLUME', 'TOTAL_AMOUNT', 'MIN_PRICE', 'MAX_PRICE',
                                     'LAST_PRICE']

        # 隐形吸筹信号
        medium_buy_groups['MEDIUM_STEALTH_ACCUMULATION'] = (
                (medium_buy_groups['VOLUME_COUNT'] >= 5) &
                (medium_buy_groups['TOTAL_VOLUME'] > self.df['VOLUME'].quantile(0.9)) &
                ((medium_buy_groups['MAX_PRICE'] - medium_buy_groups['MIN_PRICE']) / medium_buy_groups[
                    'MIN_PRICE'] < 0.01)  # 价格波动小于1%
        )

        # todo 2. 大单撤单分析（需要level2数据）
        # 这里简化处理，实际需要委托队列数据
        return medium_buy_groups

    @staticmethod
    def aggregate_analysis(df, frequency='5min', rolling_window=10):
        """按时间切片分析主力行为  为聚合分析准备数据"""
        # 重采样到指定频率
        df.set_index('TIME', inplace=True)
        resampled = df.resample(frequency).agg({
            'AMOUNT_WAN': 'sum',
            'BUY_AMOUNT': 'sum',
            'SELL_AMOUNT': 'sum',
            'LARGE_BUY': 'sum',
            'LARGE_SELL': 'sum',
            'HUGE_BUY': 'sum',
            'HUGE_SELL': 'sum',
            'PRICE': 'last',
            'IS_LARGE_ORDER': 'sum',
            'IS_HUGE_ORDER': 'sum',
            'LARGE_NET_FLOW': 'sum',
            'VOLUME': 'sum'
        })

        # 1.计算大单比例
        # count（）计算每个时间区间内非空值的数量
        resampled['MAIN_FORCE_ORDER_RATIO'] = (resampled['IS_LARGE_ORDER'] + resampled['IS_HUGE_ORDER']) / len(
            df.resample(frequency).count())
        resampled['MAIN_FORCE_BUY_RATIO'] = ((resampled['LARGE_BUY'] + resampled['HUGE_BUY'])
                                             / (resampled['LARGE_SELL'] + resampled['HUGE_SELL'] + resampled[
                    'HUGE_BUY'] + resampled['HUGE_BUY']).replace(0, 1))
        # 2.大单净流入强度
        resampled['LARGE_NET_STRENGTH'] = resampled['LARGE_NET_FLOW'] / resampled['AMOUNT_WAN']

        # 3.价格变化
        resampled['PRICE_CHANGE'] = resampled['PRICE'].pct_change(fill_method=None)
        resampled['PRICE_TREND_MA'] = resampled['PRICE_CHANGE'].rolling(rolling_window).mean()

        # 4.计算frequency的主力净流入
        resampled['MINUTE_NET_INFLOW'] = (resampled['LARGE_BUY'] + resampled['HUGE_BUY'] -
                                          resampled['LARGE_SELL'] - resampled['HUGE_SELL'])
        # 5.识别主力集中操作时段。当分钟净流入的绝对值大于其标准差时，认为主力资金活跃  识别约32%的异常活跃时段（基于正态分布假设）
        resampled['MAIN_FORCE_ACTIVE'] = resampled['MINUTE_NET_INFLOW'].abs() > resampled['MINUTE_NET_INFLOW'].std()

        # 生成交易信号
        resampled['BUY_SIGNAL'] = (
                resampled['MAIN_FORCE_ACTIVE'] &
                (resampled['MINUTE_NET_INFLOW'] > 0) &
                (resampled['MINUTE_NET_INFLOW'] > resampled['MINUTE_NET_INFLOW'].shift(1))
        # 将整个序列向后移动1个位置，相当于获取前一个时间段的数据
        )

        resampled['SELL_SIGNAL'] = (
                resampled['MAIN_FORCE_ACTIVE'] &
                (resampled['MINUTE_NET_INFLOW'] < 0) &
                (resampled['MINUTE_NET_INFLOW'] < resampled['MINUTE_NET_INFLOW'].shift(1))
        )
        return resampled

    def analyze_main_force_activity(self) -> map:
        # 拆分成 5 个子 DataFrame（行数尽量均匀）
        statistics = {
            'WINDOW_ACCUMULATION': 0,
            'WINDOW_WASH': 0,
            'WINDOW_DISTRIBUTION': 0
        }
        self.df = self.df.reset_index()  # 如果时间列是索引，先重置
        # 1. 按照时间拆分数据分片 进行主力行为分析 统计时间分片主力操作的次数
        window = 5
        chunks = self.split_by_time_window(self.df, "TIME", window)
        # 查看结果
        for i, chunk in enumerate(chunks):
            if (MainForcePatterns.identify_accumulation(chunk)):
                statistics['WINDOW_ACCUMULATION'] = statistics['WINDOW_ACCUMULATION'] + 1

            if (MainForcePatterns.identify_wash_sale(chunk)):
                statistics['WINDOW_WASH'] = statistics['WINDOW_WASH'] + 1

            if (MainForcePatterns.identify_distribution(chunk)):
                statistics['WINDOW_DISTRIBUTION'] = statistics['WINDOW_DISTRIBUTION'] + 1

        # 统计简单的时间操作的系数
        statistics['ACCUMULATION_SIMPLE_PATTERN_COUNT'] = self.df[self.df['ACCUMULATION_PATTERN'] == True].shape[0]
        statistics['DISTRIBUTION_SIMPLE_PATTERN_COUNT'] = self.df[self.df['DISTRIBUTION_PATTERN'] == True].shape[0]

        # 大单买入量
        statistics['MAIN_FORCE_BUY'] = self.df['LARGE_BUY'].sum() + self.df['HUGE_BUY'].sum()

        # 大单卖出量
        statistics['MAIN_FORCE_SELL'] = self.df['LARGE_SELL'].sum() + self.df['HUGE_SELL'].sum()

        # 主力净流入总额
        statistics['NET_MAIN_INFLOW'] = self.df['NET_MAIN_INFLOW'].sum()

        # 主力净流入强度
        total_amount = self.df['AMOUNT_WAN'].sum()
        statistics['NET_MAIN_INFLOW_RATIO'] = self.df['NET_MAIN_INFLOW'].sum() / total_amount
        main_force_total_amount = self.df['LARGE_SELL'].sum() + self.df['HUGE_SELL'].sum() + self.df['HUGE_BUY'].sum() + \
                                  self.df['HUGE_BUY'].sum()
        statistics['MAIN_FORCE_BUY_RATIO'] = ((self.df['LARGE_BUY'].sum() + self.df['HUGE_BUY'].sum())
                                              / (main_force_total_amount if main_force_total_amount != 0 else np.inf))

        # 大单买入/卖出比率
        total_buy_large = self.df['LARGE_BUY'].sum() + self.df['HUGE_BUY'].sum()
        total_sell_large = self.df['LARGE_SELL'].sum() + self.df['HUGE_SELL'].sum()
        statistics['MAIN_FORCE_BUY_SELL_RATIO'] = total_buy_large / total_sell_large if total_sell_large > 0 else float(
            'inf')

        # 主力参与度
        main_force_amount = self.df['LARGE_BUY'].sum() + self.df['LARGE_SELL'].sum() + \
                            self.df['HUGE_BUY'].sum() + self.df['HUGE_SELL'].sum()
        statistics['MAIN_FORCE_ACTIVITY_RATIO'] = main_force_amount / total_amount if total_amount > 0 else float('inf')

        # 聚合分析
        # 五分钟聚合分析
        df_5min_aggregate = self.aggregate_analysis(self.df, '5min', 10)
        df_aggregated, buy_groups = MainForcePatterns.identify_aggregate_accumulation_features(self.df.copy(),
                                                                                               df_5min_aggregate)
        statistics['MAIN_FORCE_5MIN_ACCUMULATION'] = df_aggregated['ACCUMULATION_SIGNAL_1'].astype(int).sum()
        statistics['MAIN_FORCE_5MIN_ACCUMULATION_NEAR_SUPPORT'] = df_aggregated['ACCUMULATION_SIGNAL_2'].astype(
            int).sum()
        statistics['LIKELY_MEDIUM_SPLIT_BUY'] = buy_groups["LIKELY_MEDIUM_SPLIT_BUY"].astype(int).sum()

        # 高级主力行为分析
        medium_buy_groups = self.advanced_main_force_pattern_recognition()
        statistics['MAIN_FORCE_MEDIUM_STEALTH_ACCUMULATION'] = medium_buy_groups['MEDIUM_STEALTH_ACCUMULATION'].astype(
            int).sum()

        return statistics

    @staticmethod
    def comprehensive_scoring(statistics) -> map:
        """
        综合评分系统
        """
        # 评分
        score = (
                (statistics['ACCUMULATION_SIMPLE_PATTERN_COUNT'])
                - (statistics['DISTRIBUTION_SIMPLE_PATTERN_COUNT'])
                + statistics['WINDOW_ACCUMULATION']
                + statistics['WINDOW_WASH']
                - statistics['WINDOW_DISTRIBUTION']
                + math.ceil(statistics['MAIN_FORCE_BUY'] / 5000.0) * 3
                - math.ceil(statistics['MAIN_FORCE_SELL'] / 5000.0) * 3
                + math.ceil(statistics['NET_MAIN_INFLOW'] / 3000) * 4
                + int(statistics['NET_MAIN_INFLOW_RATIO'] > 0.10) *2
                + int(statistics['MAIN_FORCE_BUY_RATIO'] > 0.6) *2
                + int(statistics['MAIN_FORCE_BUY_SELL_RATIO'] > 1)
                + int(statistics['MAIN_FORCE_ACTIVITY_RATIO'] > 0.12)
                + int(statistics['MAIN_FORCE_5MIN_ACCUMULATION'])
                + int(statistics['MAIN_FORCE_5MIN_ACCUMULATION_NEAR_SUPPORT'])
                + int(statistics['LIKELY_MEDIUM_SPLIT_BUY']) * 2
                + int(statistics['MAIN_FORCE_MEDIUM_STEALTH_ACCUMULATION']) * 2
        )
        statistics['SCORE'] = score
        return statistics

    def visualize_main_force_analysis(self):
        """可视化主力分析结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 主力净流入走势
        axes[0, 0].plot(self.df.index, self.df['NET_MAIN_INFLOW'], label='finance in')
        axes[0, 0].plot(self.df.index, self.df['MAIN_INFLOW_MA'], label='5 trade mean', linestyle='--')
        axes[0, 0].set_title('main force fiance in')
        axes[0, 0].legend()

        # 2. 大单买卖对比
        axes[0, 1].bar(self.df.index, self.df['LARGE_BUY'], label='large in', alpha=0.7)
        axes[0, 1].bar(self.df.index, -self.df['LARGE_SELL'], label='larget out', alpha=0.7)
        axes[0, 1].set_title('large in out compare')
        axes[0, 1].legend()

        # 3. 价格与主力资金关系
        ax2 = axes[1, 0].twinx()
        axes[1, 0].plot(self.df.index, self.df['PRICE'], color='red', label='price')
        ax2.bar(self.df.index, self.df['NET_MAIN_INFLOW'], alpha=0.3, label='main-force-in')
        axes[1, 0].set_title('price and main force finance')

        # 4. 主力行为模式标记
        accumulation_idx = self.df[self.df['ACCUMULATION_PATTERN']].index
        distribution_idx = self.df[self.df['DISTRIBUTION_PATTERN']].index

        axes[1, 1].plot(self.df.index, self.df['PRICE'], label='价格')
        axes[1, 1].scatter(accumulation_idx, self.df.loc[accumulation_idx, 'PRICE'],
                           color='green', label='buy_in', marker='^')
        axes[1, 1].scatter(distribution_idx, self.df.loc[distribution_idx, 'PRICE'],
                           color='red', label='sold_out', marker='v')
        axes[1, 1].set_title('main force activity model')
        axes[1, 1].legend()
        plt.tight_layout()
        plt.show()


class MainForcePatterns:
    @staticmethod
    def identify_accumulation(df_window):
        """识别吸筹模式"""
        # 条件：连续大单买入 + 价格震荡或微跌
        amount_sum = df_window['AMOUNT_WAN'].sum()
        buy_strength = (df_window['LARGE_BUY'].sum() + df_window['HUGE_BUY'].sum()) / (
            1 if amount_sum == 0 else amount_sum)
        # 变异系数。用于判断序列的波动率。越小说明越稳定
        price_volatility = df_window['PRICE'].std() / df_window['PRICE'].mean()
        return buy_strength > 0.3 and price_volatility < 0.02

    @staticmethod
    def identify_aggregate_accumulation_features(df_tick, df_aggregated, rolling_window: int = 20):
        """
        识别主力吸筹特征
        """
        # 1. 大单净流入但价格不涨或微跌
        df_aggregated['ACCUMULATION_SIGNAL_1'] = (
                (df_aggregated['LARGE_NET_FLOW'] > 0) &
                (df_aggregated['PRICE_CHANGE'].abs() < 0.005)  # 价格波动很小
        )

        # 2. 密集大单买入出现在支撑位
        df_aggregated['SUPPORT_LEVEL'] = df_aggregated['PRICE'].rolling(
            rolling_window).min()  # 用于计算时间序列中每个数据点前20个观测值（包括自身）内的最小值，也称为滚动最小值或移动最小值
        df_aggregated['NEAR_SUPPORT'] = (
                (df_aggregated['PRICE'] - df_aggregated['SUPPORT_LEVEL']) / df_aggregated['SUPPORT_LEVEL'] < 0.02
        )  # 判断是否在支撑位附近
        df_aggregated['ACCUMULATION_SIGNAL_2'] = (
                (df_aggregated['MAIN_FORCE_BUY_RATIO'] > 0.7).astype(bool) &
                df_aggregated['NEAR_SUPPORT']
        )

        # 3. 大单拆分特征（连续小单买入）
        # 识别连续买入序列
        # .diff()：计算相邻行的差值
        # != 0：判断是否发生变化（0→1或1→0）
        # .cumsum()：累积求和，为每个连续序列分配唯一分组编号
        df_tick['BUY_SEQUENCE'] = (df_tick['DIRECTION'] == 'B').astype(int)
        df_tick['BUY_GROUP'] = (df_tick['BUY_SEQUENCE'].diff() != 0).cumsum()

        # 计算连续买入组的统计
        buy_groups = df_tick[df_tick['DIRECTION'] == 'B'].groupby('BUY_GROUP').agg({
            'VOLUME': ['count', 'sum', 'mean'],
            'AMOUNT_WAN': 'sum',
            'PRICE': 'mean'
        })
        buy_groups.columns = ['VOLUME_COUNT', 'TOTAL_VOLUME', 'AVG_VOLUME', 'TOTAL_AMOUNT', 'AVG_PRICE']
        # 识别疑似拆分大单（连续多笔中等规模买入）
        buy_groups['LIKELY_MEDIUM_SPLIT_BUY'] = (
                (buy_groups['VOLUME_COUNT'] >= 3) &
                (buy_groups['AVG_VOLUME'] > df_tick['VOLUME'].quantile(0.6)) &
                (buy_groups['AVG_VOLUME'] < df_tick['VOLUME'].quantile(0.8))  # 划分成交量的大小
        )
        return df_aggregated, buy_groups

    @staticmethod
    def identify_wash_sale(df_window):
        """识别洗盘模式"""
        # 条件：大单打压 + 快速收回
        max_drawdown = (df_window['PRICE'].max() - df_window['PRICE'].min()) / df_window['PRICE'].max()
        is_drawdown_recovery = (
                    df_window["PRICE"].iloc[0] >= df_window['PRICE'].min() and df_window["PRICE"].iloc[0] <= df_window[
                'PRICE'].max())
        recovery_time = (df_window[df_window['PRICE'] == df_window['PRICE'].max()]["TIME"].astype('int64').iloc[
                             -1] // 10 ** 9 -
                         df_window[df_window['PRICE'] == df_window['PRICE'].min()]["TIME"].astype('int64').iloc[
                             0] // 10 ** 9) / 60  # 恢复时间
        # 主力净流入超过50万
        main_force_in = df_window['NET_MAIN_INFLOW'].sum() >= 0
        # 2分钟之内价格从下跌2% 并且短时间内恢复
        return is_drawdown_recovery and max_drawdown > 0.02 and 0 < recovery_time < 2 and main_force_in

    @staticmethod
    def identify_distribution(df_window):
        """识别出货模式"""
        # 条件：价格上涨 + 大单卖出
        price_increase = (df_window['PRICE'].iloc[-1] - df_window['PRICE'].iloc[0]) / df_window['PRICE'].iloc[0]
        sell_pressure = MainForcePatterns.safe_divide(df_window['LARGE_SELL'].sum() + df_window['HUGE_SELL'].sum(),
                                                      df_window['AMOUNT_WAN'].sum())
        return price_increase > 0.01 and sell_pressure > 0.4

    @staticmethod
    def safe_divide(numerator, denominator, default=0):
        """安全除法，避免除以零或NaN"""
        if denominator == 0 or pd.isna(denominator) or pd.isna(numerator):
            return default
        return numerator / denominator


def analyze_stocks(stocks: pd.DataFrame, file: TextIOWrapper, symbol: str, need_save: bool = False,
                   date_str: str = '2025-09-30'):
    ts.set_token("5b03bbe59725c145f229d4eb7fe73d8fc9dc98d9cde5e194a0e4d708")
    print(f"获取到了 {symbol}  {stocks.shape[0]}  只股票")
    current_dir = os.path.dirname(os.path.abspath(__file__))
    res_df = pd.DataFrame(columns=[
        'CODE', 'NAME', 'SCORE', 'ACCUMULATION_SIMPLE_PATTERN_COUNT', 'DISTRIBUTION_SIMPLE_PATTERN_COUNT',
        'WINDOW_ACCUMULATION', 'WINDOW_WASH', 'WINDOW_DISTRIBUTION', 'MAIN_FORCE_BUY',
        'MAIN_FORCE_SELL', 'NET_MAIN_INFLOW', 'NET_MAIN_INFLOW_RATIO', 'MAIN_FORCE_BUY_RATIO',
        'MAIN_FORCE_BUY_SELL_RATIO', 'MAIN_FORCE_ACTIVITY_RATIO', 'MAIN_FORCE_5MIN_ACCUMULATION',
        'MAIN_FORCE_5MIN_ACCUMULATION_NEAR_SUPPORT',
        'LIKELY_MEDIUM_SPLIT_BUY', 'MAIN_FORCE_MEDIUM_STEALTH_ACCUMULATION'
    ])

    # 东财数据
    for index in tqdm(range(stocks.shape[0])):
        row = stocks.iloc[index]
        if index <= 6000:
            try:
                df = None
                if need_save:
                    df = ts.realtime_tick(ts_code=f'{row["证券代码"]}.{symbol}', src='dc')
                    df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce")
                    df["PRICE"] = df["PRICE"].astype('float64')
                    df['AMOUNT'] = df["VOLUME"] * df["PRICE"]
                    tick_data_path = f'{current_dir}/data/{symbol}/{datetime.now().date()}'
                    # 检查文件夹是否存在，不存在则创建
                    if not os.path.exists(tick_data_path):
                        os.makedirs(tick_data_path)  # 递归创建目录（包括父目录）
                    df.to_csv(f'{tick_data_path}/{row["证券代码"]}_{symbol}_{row["证券简称"]}.csv')
                else:
                    df = pd.read_csv(
                        f'{current_dir}/data/{symbol}/{date_str}/{str(row["证券代码"]).zfill(6)}_{symbol}_{row["证券简称"]}.csv')

                analyzer = MainForceAnalyzer(df)
                statistics = analyzer.analyze_main_force_activity()
                statistics = MainForceAnalyzer.comprehensive_scoring(statistics)

                statistics['CODE'] = row["证券代码"]
                statistics['NAME'] = row["证券简称"]
                res_df.loc[len(res_df)] = statistics
                if index % 1000 == 0:
                    file.flush()
            except Exception as e:
                print(f'处理{row["证券代码"]}.{symbol} 发生异常！')
                traceback.print_exc()  # 打印完整的堆栈跟踪
                continue

    res_df.to_csv(file, index=False)


def get_stock_name_code(path: str, symbol: str) -> pd.DataFrame:
    if path is not None and path.strip() != "":
        file_list = [entry.name.replace("_", ",") for entry in os.scandir(path) if entry.is_file()]
        file_list = [entry.replace(".", ",") for entry in file_list]
        # 示例数据：字符串列表（每行是逗号分隔的值）
        header = "证券代码,symbol,证券简称,file_type"
        file_list.insert(0, header)
        # 合并成字符串并用 StringIO 模拟文件对象
        csv_string = "\n".join(file_list)
        df = pd.read_csv(StringIO(csv_string))
        return df
    else:
        if symbol == "SH":
            stock_sh = stock_info_sh_name_code(symbol="主板A股")
            stock_sh = stock_sh[["证券代码", "证券简称"]]
            return stock_sh
        elif symbol == "SZ":
            # 分析深圳
            stock_sz = stock_info_sz_name_code(symbol="A股列表")
            stock_sz["证券代码"] = stock_sz["A股代码"].astype(str).str.zfill(6)
            stock_sz["证券简称"] = stock_sz["A股简称"]
            return stock_sz
    return None


def analyze_tick_data(date_str: str, need_save: bool = False, ) -> pd.DataFrame:
    """
    沪深京 A 股列表
    :return: 沪深京 A 股数据
    :rtype: pandas.DataFrame
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    with open(f"{current_dir}/result/{date_str}_output_sh.csv", "w") as f:
        # # 分析上证
        if not need_save:
            stock_sh = get_stock_name_code(f"{current_dir}/data/SH/{date_str}", "SH")
            analyze_stocks(stock_sh, f, symbol="SH", need_save=False, date_str=date_str)
        else:
            # 分析上证
            stock_sh = get_stock_name_code(None, "SH")
            analyze_stocks(stock_sh, f, symbol="SH", need_save=True)

    with open(f"{current_dir}/result/{date_str}_output_sz.csv", "w") as f:
        if not need_save:
            # 分析深圳
            stock_sz = get_stock_name_code(f"{current_dir}/data/SZ/{date_str}", "SZ")
            analyze_stocks(stock_sz, f, symbol="SZ", need_save=False, date_str=date_str)
        else:
            # 分析深圳
            stock_sz = get_stock_name_code(None, "SZ")
            analyze_stocks(stock_sz, f, symbol="SZ", need_save=True)

    # 科创板
    # stock_kcb = stock_info_sh_name_code(symbol="科创板")
    # stock_kcb = stock_kcb[["证券代码", "证券简称"]]

    # 北交所
    # stock_bse = stock_info_bj_name_code()
    # stock_bse = stock_bse[["证券代码", "证券简称"]]
    # stock_bse.columns = ["证券代码", "证券简称"]
    #
    # big_df = pd.concat(objs=[big_df, stock_sh], ignore_index=True)
    # big_df = pd.concat(objs=[big_df, stock_kcb], ignore_index=True)
    # big_df = pd.concat(objs=[big_df, stock_bse], ignore_index=True)
    # big_df.columns = ["code", "name"]


if __name__ == '__main__':
    analyze_tick_data('2025-10-08', False)
