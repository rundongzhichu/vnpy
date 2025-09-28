from functools import lru_cache

import akshare as ak
import numpy as np
import pandas as pd
import requests

# Tushare（中国股市专用）：提供A股、港股的逐笔成交、订单簿数据，需注册获取Token
import tushare as ts
from akshare import stock_info_sz_name_code, stock_info_bj_name_code, stock_info_sh_name_code
from matplotlib import pyplot as plt

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

    def analyze(self):
        """执行完整的主力分析"""
        self.df["TIME"] = pd.to_datetime(self.df['TIME'])
        self.calculate_order_size()
        self.classify_orders()
        self.identify_main_force_patterns()

    def calculate_order_size(self):
        """计算每笔成交的订单规模"""
        # 计算每笔成交的金额（万元）
        self.df['AMOUNT_WAN'] = self.df['AMOUNT'] / 10000

        # 定义大单阈值（可根据股票流通盘调整）
        self.df['IS_LARGE_ORDER'] = self.df['AMOUNT_WAN'] > 50  # 50万元以上的算大单
        self.df['IS_HUGE_ORDER'] = self.df['AMOUNT_WAN'] > 200  # 200万元以上的算特大单

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

        self.df['BUY_AMOUNT'] = np.where(buy_condition, self.df['AMOUNT_WAN'], 0)
        self.df['SELL_AMOUNT'] = np.where(sell_condition, self.df['AMOUNT_WAN'], 0)
        self.df['NEUTRAL_TRADING_VOLUME'] = np.where(neutral_condition, self.df['AMOUNT_WAN'], 0)

        # 计算10笔交易的价格变化
        self.df['PRICE_CHANGE_10_TRADE'] = self.df['PRICE'].pct_change(periods=10).fillna(0).replace([np.inf, -np.inf], 0) * 100 >= 0
        # 大单资金流向
        self.df['LARGE_BUY'] = np.where((buy_condition | (neutral_condition & self.df['PRICE_CHANGE_10_TRADE'] )) & self.df['IS_LARGE_ORDER'],
                                        self.df['AMOUNT_WAN'], 0)
        self.df['LARGE_SELL'] = np.where(sell_condition | (neutral_condition & self.df['PRICE_CHANGE_10_TRADE']) & self.df['IS_LARGE_ORDER'],
                                         self.df['AMOUNT_WAN'], 0)

        # 特大单资金流向
        self.df['HUGE_BUY'] = np.where(buy_condition & self.df['IS_HUGE_ORDER'],
                                       self.df['AMOUNT_WAN'], 0)
        self.df['HUGE_SELL'] = np.where(sell_condition & self.df['IS_HUGE_ORDER'],
                                        self.df['AMOUNT_WAN'], 0)

    def identify_main_force_patterns(self):
        """识别主力操作模式"""
        # 计算滚动窗口内的主力行为
        window = 5  # 5笔成交为一个分析窗口

        # 主力净流入
        self.df['NET_MAIN_INFLOW'] = (self.df['LARGE_BUY'] + self.df['HUGE_BUY'] -
                                      self.df['LARGE_SELL'] - self.df['HUGE_SELL'])
        # 主力净流入滚动均值
        self.df['MAIN_INFLOW_MA'] = self.df['NET_MAIN_INFLOW'].rolling(window=window).mean()

        # 识别主力吸筹模式（价格平稳或下跌时大单买入）
        self.df['PRICE_CHANGE'] = self.df['PRICE'].pct_change() * 100
        self.df['ACCUMULATION_PATTERN'] = (
                (self.df['NET_MAIN_INFLOW'] > 0) &
                (self.df['PRICE_CHANGE'] < 0.5)  # 价格涨幅小于0.5%
        )
        # 识别主力出货模式（价格上涨时大单卖出）
        self.df['DISTRIBUTION_PATTERN'] = (
                (self.df['NET_MAIN_INFLOW'] < 0) &
                (self.df['PRICE_CHANGE'] > 0.5)  # 价格涨幅大于0.5%
        )

    @staticmethod
    def split_to_windows(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
        """ 按照交易记录条数拆分数据到指定窗口大小 """
        for i in range(0, len(df), window_size):
            yield i / window_size, df.iloc[i: i + window_size]

    @staticmethod
    def split_by_time_window(df, time_col, window=3):
        """按时间窗口拆分 DataFrame  默认三分钟"""
        df[time_col] = pd.to_datetime(df[time_col])  # 确保是 datetime
        # df = df.sort_values(time_col)  # 按时间排序

        # 计算分组（每 3 分钟一组）
        df["group"] = ((df[time_col].astype('int64') // 10 ** 9).astype('int64') // (np.int64(window) * 60)).astype(
            "int")
        # 按分组拆分
        chunks = [group for _, group in df.groupby("group")]
        # 删除临时分组列
        for chunk in chunks:
            chunk.drop("group", axis=1, inplace=True)
        return chunks

    def analyze_main_force_activity(self) -> map:
        # 拆分成 5 个子 DataFrame（行数尽量均匀）
        statistics = {
            'ACCUMULATION': 0,
            'WASH': 0
        }
        self.df = self.df.reset_index()  # 如果时间列是索引，先重置
        # 按照时间拆分数据分片
        chunks = self.split_by_time_window(self.df, "TIME", 3)
        # 查看结果
        for i, chunk in enumerate(chunks):
            if (MainForcePatterns.identify_accumulation(chunk)):
                statistics['ACCUMULATION'] = statistics['ACCUMULATION'] + 1

            if (MainForcePatterns.identify_wash_sale(chunk)):
                statistics['WASH'] = statistics['WASH'] + 1

        statistics['主力吸筹次数'] = self.df[self.df['ACCUMULATION_PATTERN'] == True].shape[0]
        statistics['主力派发筹码次数'] = self.df[self.df['DISTRIBUTION_PATTERN'] == True].shape[0]

        # 主力参与度
        total_amount = self.df['AMOUNT_WAN'].sum()
        main_force_amount = self.df['LARGE_BUY'].sum() + self.df['LARGE_SELL'].sum() + \
                            self.df['HUGE_BUY'].sum() + self.df['HUGE_SELL'].sum()
        statistics['主力参与度'] = main_force_amount / total_amount if total_amount > 0 else 0

        # 主力净流入强度
        statistics['主力净流入强度'] = (self.df['LARGE_BUY'].sum() + self.df['HUGE_BUY'].sum() -
                                    self.df['LARGE_SELL'].sum() - self.df['HUGE_SELL'].sum()) / total_amount

        # 大单买入/卖出比率
        total_buy_large = self.df['LARGE_BUY'].sum() + self.df['HUGE_BUY'].sum()
        total_sell_large = self.df['LARGE_SELL'].sum() + self.df['HUGE_SELL'].sum()
        statistics['大单买卖比例'] = total_buy_large / total_sell_large if total_sell_large > 0 else float('inf')

        return statistics

    def analyze_by_time_slice(self, frequency='3min'):
        """按时间切片分析主力行为"""
        # 重采样到指定频率
        self.df.set_index('TIME', inplace=True)
        resampled = self.df.resample(frequency).agg({
            'AMOUNT_WAN': 'sum',
            'BUY_AMOUNT': 'sum',
            'SELL_AMOUNT': 'sum',
            'LARGE_BUY': 'sum',
            'LARGE_SELL': 'sum',
            'HUGE_BUY': 'sum',
            'HUGE_SELL': 'sum',
            'PRICE': 'last'
        })

        # 计算每分钟的主力净流入
        resampled['MINUTE_NET_INFLOW'] = (resampled['LARGE_BUY'] + resampled['HUGE_BUY'] -
                                          resampled['LARGE_SELL'] - resampled['HUGE_SELL'])
        # 识别主力集中操作时段
        resampled['MAIN_FORCE_ACTIVE'] = resampled['MINUTE_NET_INFLOW'].abs() > resampled['MINUTE_NET_INFLOW'].std()
        return resampled

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
        buy_strength = df_window['LARGE_BUY'].sum() / df_window['AMOUNT_WAN'].sum()
        # 变异系数。用于判断序列的波动率。越小说明越稳定
        price_volatility = df_window['PRICE'].std() / df_window['PRICE'].mean()
        return buy_strength > 0.3 and price_volatility < 0.02

    @staticmethod
    def identify_wash_sale(df_window):
        """识别洗盘模式"""
        # 条件：大单打压 + 快速收回
        max_drawdown = (df_window['PRICE'].max() - df_window['PRICE'].min()) / df_window['PRICE'].max()
        is_drawdown_recovery = (df_window["PRICE"].iloc[0] >= df_window['PRICE'].min() and df_window["PRICE"].iloc[0] <= df_window['PRICE'].max())
        recovery_time = (df_window[df_window['PRICE'] == df_window['PRICE'].max()]["TIME"].astype('int64').iloc[-1] // 10**9 -
                         df_window[df_window['PRICE'] == df_window['PRICE'].min()]["TIME"].astype('int64').iloc[0] // 10**9) / 60  # 恢复时间
        # 主力净流入超过50万
        main_force_in = df_window['NET_MAIN_INFLOW'].sum() >= 0
        # 1分钟之内价格从下跌2% 并且短时间内恢复
        return is_drawdown_recovery and max_drawdown > 0.02 and 0 < recovery_time < 2 and main_force_in

    @staticmethod
    def identify_distribution(df_window):
        """识别出货模式"""
        # 条件：价格上涨 + 大单卖出
        price_increase = (df_window['PRICE'].iloc[-1] - df_window['PRICE'].iloc[0]) / df_window['PRICE'].iloc[0]
        sell_pressure = df_window['LARGE_SELL'].sum() / df_window['AMOUNT_WAN'].sum()
        return price_increase > 0.01 and sell_pressure > 0.4

def analyze_stocks(stocks: pd.DataFrame, output_file: str, mode:str, symbol: str):
    ts.set_token("5b03bbe59725c145f229d4eb7fe73d8fc9dc98d9cde5e194a0e4d708")
    with open(output_file, mode) as f:
        # 东财数据
        for index, row in stocks.iterrows():
            if index <= 6000:
                try:
                    df = ts.realtime_tick(ts_code=f'{row["证券代码"]}.{symbol}', src='dc')
                    df["VOLUME"] = pd.to_numeric(df["VOLUME"], errors="coerce")
                    df["PRICE"] = df["PRICE"].astype('float64')
                    df['AMOUNT'] = df["VOLUME"] * df["PRICE"]
                    analyzer = MainForceAnalyzer(df)
                    statistics = analyzer.analyze_main_force_activity()
                    print(f'获取到了 证券：{row["证券代码"]}.{symbol} {row["证券简称"]} 主力监控信息 ' + str(statistics), file=f)
                    if statistics['ACCUMULATION'] > 10 and statistics['WASH'] > 5 \
                            and analyzer.df[analyzer.df['ACCUMULATION_PATTERN'] == True].shape[0] > 100:
                        print(f'获取到了 证券：{row["证券代码"]}.{symbol} {row["证券简称"]} 多次出现主力吸筹和洗盘', file=f)
                except Exception as e:
                    print(e.with_traceback(), file=f)
                    continue


@lru_cache()
def stock_info_a_code_name() -> pd.DataFrame:
    """
    沪深京 A 股列表
    :return: 沪深京 A 股数据
    :rtype: pandas.DataFrame
    """
    # 分析上证
    stock_sh = stock_info_sh_name_code(symbol="主板A股")
    stock_sh = stock_sh[["证券代码", "证券简称"]]
    analyze_stocks(stock_sh, output_file="output.txt", mode="w", symbol="SH")

    # 分析深圳
    stock_sz = stock_info_sz_name_code(symbol="A股列表")
    stock_sz["证券代码"] = stock_sz["A股代码"].astype(str).str.zfill(6)
    stock_sz["证券简称"] = stock_sz["A股简称"]
    analyze_stocks(stock_sh, output_file="output.txt", mode="w", symbol="SZ")

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


stock_info_a_code_name()