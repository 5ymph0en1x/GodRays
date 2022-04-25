import os
import pandas as pd
from datetime import datetime, timedelta, timezone
import finplot as fplt
from logzero import logger
from math import nan
import math
import matplotlib.dates as mpl_dates
import peakdetect
from PyQt5.QtWidgets import QComboBox, QWidget, QApplication, QPushButton, QProgressBar, QLabel, QLineEdit
import pyqtgraph as pg
from shapely.geometry import LineString
import time
import pytz
import pybitmex.bitmex_http_com as bitmex
from colorama import Fore, init
import warnings
from math import atan2, degrees
from pyqtgraph import QtCore, QtGui
import sys


# tol = 0.085  # 0.1 # Tolerance in $ for Triple Crosses Detection
# tol2 = 0.000025  # 0.00005 # Tolerance for Parallelism Detection
offset_graph = 250

client_bm = bitmex.bitmex(test=False, api_key='', api_secret='')
MAX_TABLE_LEN = 1000
warnings.filterwarnings("ignore")
init(autoreset=True)
os.environ['NUMEXPR_MAX_THREADS'] = str(6)
os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"
CONFIG_PATH = "config.yml"
QApplication.setHighDpiScaleFactorRoundingPolicy(QtCore.Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
QtCore.QLocale.setDefault(QtCore.QLocale(QtCore.QLocale.C))


def findItemByKeys(keys, table, matchData):
    for item in table:
        matched = True
        for key in keys:
            if item[key] != matchData[key]:
                matched = False
        if matched:
            return item


def order_leaves_no_quantity(o):
    if o['leavesQty'] is None or o['leavesQty'] == 0:
        return True
    else:
        return False


class Live_Analysis:
    def __init__(self):
        self.symbol = None
        self.history = 0
        self.df = pd.DataFrame()
        self.df_time = pd.DataFrame()
        self.minute_cached = None
        self.offset_cached = 0
        self.first = 0

    def get_index_reference(self, df_time, coo):
        j = 0
        for i in range(len(df_time)):
            if df_time.Time[i] == coo:
                j = i
                break
            elif i == (len(df_time) - 1):
                # print('No match: ', df_time.Time[i], ' and coo: ', coo)
                return None
            else:
                continue
        return j

    def draw(self, value, df_time, type, coo, ax, color):
        pen = pg.mkPen(color=color, style=QtCore.Qt.SolidLine)
        if type == 'v':
            j = 0
            for i in range(len(df_time)):
                if df_time.Time[i] == coo:
                    j = i
                    break
                elif i >= (len(df_time) - 1):
                    # print('No match: ', df_time.Time[i], ' and coo: ', coo)
                    return
                else:
                    continue
            vline = pg.InfiniteLine(movable=False, pen=pen)
            vline.setAngle(90)
            vline.setValue(j)
            ax.addItem(vline, ignoreBounds=False)

        if type == 'h':
            hline = pg.InfiniteLine(movable=False, pen=pen)
            hline.setAngle(0)
            hline.setValue(value)
            ax.addItem(hline, ignoreBounds=False)

    def do_load_price_history(self, period):
        try:
            df = pd.DataFrame(index=range(int(self.history)), columns=['Time', 'Open', 'Close', 'High', 'Low', 'Volume'])
            df_time = pd.DataFrame(index=range(int(self.history)), columns=['Time'])
            data = []
            k = self.history
            iterations = math.ceil(k / 750)
            time_starter = [0] * (iterations + 1)

            for i in range(iterations):
                if period == '1m':
                    time_starter[i] = datetime.utcnow() - timedelta(minutes=k)
                    k -= 750
                elif period == '5m':
                    time_starter[i] = datetime.utcnow() - timedelta(minutes=k * 5)
                    k -= 750
                elif period == '1h':
                    time_starter[i] = datetime.utcnow() - timedelta(minutes=k * 60)
                    k -= 750
                else:
                    time_starter[i] = datetime.utcnow() - timedelta(minutes=k * 1440)
                    k -= 750

            for i in range(iterations):
                if period == '1m':
                    data_temp = client_bm.Trade.Trade_getBucketed(symbol=self.symbol, binSize='1m', count=750, partial=True,
                                                                  startTime=time_starter[i]).result()
                elif period == '5m':
                    data_temp = client_bm.Trade.Trade_getBucketed(symbol=self.symbol, binSize='5m', count=750, partial=True,
                                                                  startTime=time_starter[i]).result()
                elif period == '1h':
                    data_temp = client_bm.Trade.Trade_getBucketed(symbol=self.symbol, binSize='1h', count=750, partial=True,
                                                                  startTime=time_starter[i]).result()
                else:
                    data_temp = client_bm.Trade.Trade_getBucketed(symbol=self.symbol, binSize='1d', count=750, partial=True,
                                                                  startTime=time_starter[i]).result()
                data.append(data_temp[0])

            j = 0
            k = self.history
            for l in range(iterations):
                if j >= k:
                    break
                for i in data[l]:
                    df.loc[j] = pd.Series(
                        {'Time': i['timestamp'], 'Open': i['open'], 'Close': i['close'], 'High': i['high'], 'Low': i['low'], 'Volume': i['volume']})
                    df_time.Time.loc[j] = i['timestamp']
                    j += 1
                    if j >= k:
                        break

            df_proxy = pd.DataFrame(index=range(self.history, self.history + int(offset_graph), 1), columns=['Time', 'Open', 'Close', 'High', 'Low', 'Volume'])
            for i in range(self.history, self.history + int(offset_graph), 1):
                df_proxy.Time.loc[i] = i
            df = df.append(df_proxy)

            start_date = df_time.Time[self.history - 1]
            if period == '1m':
                proxy = pd.to_datetime(start_date + timedelta(minutes=int(offset_graph)))
            elif period == '5m':
                proxy = pd.to_datetime(start_date + timedelta(minutes=int(offset_graph) * 5))
            elif period == '1h':
                proxy = pd.to_datetime(start_date + timedelta(minutes=int(offset_graph) * 60))
            else:
                proxy = pd.to_datetime(start_date + timedelta(minutes=int(offset_graph) * 1440))
            table = pd.DataFrame(index=range(1), columns=['Time'])
            table.Time.loc[0] = proxy
            df_time = df_time.append(table)
            df_time.set_index('Time', inplace=True)
            if period == '1m':
                df_time = df_time.asfreq('1T')
            elif period == '5m':
                df_time = df_time.asfreq('5T')
            elif period == '1h':
                df_time = df_time.asfreq('60T')
            else:
                df_time = df_time.asfreq('1440T')
            df_time = df_time.reset_index(drop=False)

            return df, df_time

        except Exception as e:
            logger.exception(e)
            time.sleep(60)
            pass

    def calc_parabolic_sar(self, df, af=0.2, steps=10):
        up = True
        sars = [nan] * len(df)
        sar = ep_lo = df.Low.iloc[0]
        ep = ep_hi = df.High.iloc[0]
        aaf = af
        aaf_step = aaf / steps
        af = 0
        for i,(hi,lo) in enumerate(zip(df.High, df.Low)):
            # parabolic sar formula:
            sar = sar + af * (ep - sar)
            # handle new extreme points
            if hi > ep_hi:
                ep_hi = hi
                if up:
                    ep = ep_hi
                    af = min(aaf, af+aaf_step)
            elif lo < ep_lo:
                ep_lo = lo
                if not up:
                    ep = ep_lo
                    af = min(aaf, af+aaf_step)
            # handle switch
            if up:
                if lo < sar:
                    up = not up
                    sar = ep_hi
                    ep = ep_lo = lo
                    af = 0
            else:
                if hi > sar:
                    up = not up
                    sar = ep_lo
                    ep = ep_hi = hi
                    af = 0
            sars[i] = sar
        df['sar'] = sars
        return df['sar']

    def calc_rsi(self, price, n=14, ax=None):
        diff = price.diff().values
        gains = diff
        losses = -diff
        gains[~(gains>0)] = 0.0
        losses[~(losses>0)] = 1e-10 # we don't want divide by zero/NaN
        m = (n-1) / n
        ni = 1 / n
        g = gains[n] = gains[:n].mean()
        l = losses[n] = losses[:n].mean()
        gains[:n] = losses[:n] = nan
        for i,v in enumerate(gains[n:],n):
            g = gains[i] = ni*v + m*g
        for i,v in enumerate(losses[n:],n):
            l = losses[i] = ni*v + m*l
        rs = gains / losses
        rsi = 100 - (100/(1+rs))
        return rsi

    def calc_stochastic_oscillator(self, df, n=14, m=3, smooth=3):
        lo = df.Low.rolling(n).min()
        hi = df.High.rolling(n).max()
        k = 100 * (df.Close-lo) / (hi-lo)
        d = k.rolling(m).mean()
        return k, d

    def calc_plot_data(self, df, df_time, indicators):
        '''Returns data for all plots and for the price line.'''
        price = df['Open Close High Low'.split()]
        price.index = df_time.Time

        volume = df['Open Close Volume'.split()]
        volume.index = df_time.Time

        ma50 = ma200 = vema24 = sar = rsi = stoch = stoch_s = None
        if 'few' in indicators or 'moar' in indicators:
            ma50  = price.Close.rolling(50).mean()
            ma200 = price.Close.rolling(200).mean()
            vema24 = volume.Volume.ewm(span=24).mean()
        if 'moar' in indicators:
            sar = self.calc_parabolic_sar(df)
            rsi = self.calc_rsi(df.Close)
            stoch,stoch_s = self.calc_stochastic_oscillator(df)
        plot_data = dict(price=price, volume=volume, ma50=ma50, ma200=ma200, vema24=vema24, sar=sar, rsi=rsi, stoch=stoch, stoch_s=stoch_s)
        # for price line
        last_close = price.iloc[self.history - 1].Close
        if last_close is not None and price.iloc[self.history-2].Close is not None:
            last_col = fplt.candle_bull_color if last_close > price.iloc[self.history-2].Close else fplt.candle_bear_color
        else:
            last_col = '#4b8'
        price_data = dict(last_close=last_close, last_col=last_col)

        return plot_data, price_data

    def update_data(self):
        try:
            data = []
            period = ctrl_panel.period.currentText()

            real_utc_now = datetime.utcnow().replace(tzinfo=pytz.utc)  # .replace(microsecond=0)
            server_time = self.df_time.Time[self.history-1].replace(tzinfo=pytz.utc)  # .replace(microsecond=0)

            offset = server_time - real_utc_now
            offset = offset.seconds
            offset = (offset % 3600) // 60

            if period == '1m':
                offset_processed = offset + 10
                data_temp = client_bm.Trade.Trade_getBucketed(symbol=self.symbol, binSize='1m', count=offset_processed,
                                                              partial=True, reverse=True).result()
            elif period == '5m':
                offset_processed = math.ceil(offset / 5) + 5
                data_temp = client_bm.Trade.Trade_getBucketed(symbol=self.symbol, binSize='5m', count=offset_processed,
                                                              partial=True, reverse=True).result()
            elif period == '1h':
                offset_processed = math.ceil(offset / 60) + 1
                data_temp = client_bm.Trade.Trade_getBucketed(symbol=self.symbol, binSize='1h', count=offset_processed,
                                                              partial=True, reverse=True).result()
            else:
                offset_processed = math.ceil(offset / 1440) + 1
                data_temp = client_bm.Trade.Trade_getBucketed(symbol=self.symbol, binSize='1d', count=offset_processed,
                                                              partial=True, reverse=True).result()

            if server_time.timestamp() >= real_utc_now.timestamp():
                for i in range(offset_processed):
                    data.append(data_temp[0][i])
                df_temp = pd.DataFrame(index=range(offset_processed),
                                       columns=['Time', 'Open', 'Close', 'High', 'Low', 'Volume'])
                for i in range(0, offset_processed, 1):
                    df_temp.loc[i] = pd.Series(
                        {'Time': data[i]['timestamp'], 'Open': data[i]['open'], 'Close': data[i]['close'],
                         'High': data[i]['high'],
                         'Low': data[i]['low'], 'Volume': data[i]['volume']})
                df_temp['Time'] = df_temp['Time'].apply(lambda d: pd.to_datetime(d))

                self.df_time = pd.DataFrame(data=self.df_time.Time, columns=['Time'])
                # self.df_time = self.df_time.shift(periods=(offset_processed * -1))

                df_temp = df_temp[::-1]
                self.df = self.df[:self.history - len(df_temp)]
                self.df = self.df.append(df_temp)
                df_proxy = pd.DataFrame(index=range(self.history, self.history + int(offset_graph)),
                                        columns=['Time', 'Open', 'Close', 'High', 'Low', 'Volume'])
                for i in range(self.history, self.history + len(df_proxy), 1):
                    df_proxy.Time.loc[i] = i
                self.df = self.df.append(df_proxy)

                start_date = self.df_time.Time[0]
                if period == '1m':
                    self.df_time.Time = pd.date_range(start_date, periods=self.history+offset_graph, freq='1T')
                elif period == '5m':
                    self.df_time.Time = pd.date_range(start_date, periods=self.history + offset_graph, freq='5T')
                elif period == '1h':
                    self.df_time.Time = pd.date_range(start_date, periods=self.history + offset_graph, freq='60T')
                else:
                    self.df_time.Time = pd.date_range(start_date, periods=self.history + offset_graph, freq='1440T')
                self.df_time.set_index('Time', inplace=True)
                self.df_time = self.df_time.reset_index(drop=False)
                self.df = self.df.reset_index(drop=True)
                self.df.Time = self.df.index
                # print(self.df[:history + 3].tail(10))

                return self.df, self.df_time

            else:
                # print('type 2: ', server_time)

                for i in range(offset_processed):
                    data.append(data_temp[0][i])
                df_temp = pd.DataFrame(index=range(offset_processed), columns=['Time', 'Open', 'Close', 'High', 'Low', 'Volume'])
                for i in range(offset_processed):
                    df_temp.loc[i] = pd.Series({'Time': data[i]['timestamp'], 'Open': data[i]['open'], 'Close': data[i]['close'], 'High': data[i]['high'],
                                               'Low': data[i]['low'], 'Volume': data[i]['volume']})
                self.df_time = pd.DataFrame(data=self.df_time.Time, columns=['Time'])
                self.df_time = self.df_time.shift(periods=(-1))
                proxy_date = pd.date_range((self.df_time.Time[len(self.df_time) - 2] + timedelta(minutes=1)).to_numpy(), periods=1, freq='1T').to_pydatetime().tolist()
                proxy_date = proxy_date[0].timestamp()
                self.df_time.Time[len(self.df_time) - 1] = datetime.fromtimestamp(proxy_date).strftime('%Y-%m-%d %H:%M:%S.%f')

                df_temp = df_temp[::-1]
                self.df = self.df[:self.history - offset_processed + 1]
                # df_ = df_.reset_index(drop=True)
                self.df = self.df.append(df_temp)
                df_proxy = pd.DataFrame(index=range(self.history, self.history + int(offset_graph)),
                                        columns=['Time', 'Open', 'Close', 'High', 'Low', 'Volume'])
                for i in range(self.history, self.history + int(offset_graph), 1):
                    df_proxy.Time.loc[i] = i
                self.df = self.df.append(df_proxy)
                self.df = self.df.shift(periods=(-1))
                self.df = self.df[:-1]
                self.df = self.df.reset_index(drop=True)
                self.df.Time = self.df.index

                return self.df, self.df_time

        except Exception as e:
            logger.exception(e)
            time.sleep(60)
            self.df = [0]
            self.df_time = [0]
            return self.df, self.df_time

    def realtime_update_plot(self):
        '''Called at regular intervals by a timer.'''

        # calculate the new plot data
        indicators = ctrl_panel.indicators.currentText().lower()
        period = ctrl_panel.period.currentText()

        if self.first <= 1:
            self.df, self.df_time = self.do_load_price_history(period)
            self.first += 1
            return

        self.df, self.df_time = self.update_data()
        if len(self.df) <= 1 or len(self.df_time) <= 1:
            self.first = 1
            return
        data, price_data = self.calc_plot_data(self.df, self.df_time, indicators)

        elapsed_sec = datetime.utcnow().second
        ctrl_panel.time.setText(('-> ' + str(elapsed_sec) + 's'))

        # first update all data, then graphics (for zoom rigidity)
        for k in data:
            if data[k] is not None:
                plots[k].update_data(data[k], gfx=False)
        for k in data:
            if data[k] is not None:
                plots[k].update_gfx()

        # place and color price line
        ax.price_line.setPos(price_data['last_close'])
        ax.price_line.pen.setColor(pg.mkColor(price_data['last_col']))

    def load_asset(self, period):
        '''Resets and recalculates everything, and plots for the first time.'''
        # save window zoom position before resetting
        fplt._savewindata(fplt.windows[0])

        self.symbol = ctrl_panel.symbol.currentText()
        self.history = int(ctrl_panel.history.text())
        # ws.close()
        df, df_time = self.do_load_price_history(period)
        # ws.reconnect(symbol, interval, self.df)

        # remove any previous plots
        ax.reset()
        axo.reset()
        ax_rsi.reset()

        # calculate plot data
        indicators = ctrl_panel.indicators.currentText().lower()
        data, price_data = self.calc_plot_data(df, df_time, indicators)

        # some space for legend
        ctrl_panel.move(100 if 'clean' in indicators else 200, 0)

        # plot data
        global plots
        plots = {}
        plots['price'] = fplt.candlestick_ochl(data['price'], ax=ax)
        plots['volume'] = fplt.volume_ocv(data['volume'], ax=axo)
        if data['ma50'] is not None:
            plots['ma50'] = fplt.plot(data['ma50'], legend='MA-50', ax=ax)
            plots['ma200'] = fplt.plot(data['ma200'], legend='MA-200', ax=ax)
            plots['vema24'] = fplt.plot(data['vema24'], color=4, legend='V-EMA-24', ax=axo)
        if data['rsi'] is not None:
            ax.set_visible(xaxis=False)
            ax_rsi.show()
            fplt.set_y_range(0, 100, ax=ax_rsi)
            fplt.add_band(30, 70, color='#6335', ax=ax_rsi)
            plots['sar'] = fplt.plot(data['sar'], color='#55a', style='+', width=0.6, legend='SAR', ax=ax)
            plots['rsi'] = fplt.plot(data['rsi'], legend='RSI', ax=ax_rsi)
            plots['stoch'] = fplt.plot(data['stoch'], color='#880', legend='Stoch', ax=ax_rsi)
            plots['stoch_s'] = fplt.plot(data['stoch_s'], color='#650', ax=ax_rsi)
        else:
            ax.set_visible(xaxis=True)
            ax_rsi.hide()
        # price line
        ax.price_line = pg.InfiniteLine(angle=0, movable=False, pen=fplt._makepen(fplt.candle_bull_body_color, style='.'))
        ax.price_line.setPos(price_data['last_close'])
        ax.price_line.pen.setColor(pg.mkColor(price_data['last_col']))
        ax.addItem(ax.price_line, ignoreBounds=True)

        # restores saved zoom position, if in range
        fplt.refresh()
        return df, df_time

    def create_ctrl_panel(self, win):
        panel = QWidget(win)
        panel.move(100, 0)
        win.scene().addWidget(panel)
        layout = QtGui.QGridLayout(panel)

        def grab_screenshot():
            time_ToPrint = str(datetime.utcnow().strftime('%y-%m-%d_%H-%M'))
            name_ToPrint = "GodRays_" + time_ToPrint + '.png'
            filename = QtGui.QFileDialog.getSaveFileName(panel, 'Save File', name_ToPrint, '*.png')
            QtGui.QScreen.grabWindow(QApplication.primaryScreen(), win.winId()).save(filename[0], 'PNG')

        panel.symbol = QComboBox(panel)
        [panel.symbol.addItem(i+'USD') for i in 'XBT ETH'.split()]
        panel.symbol.setCurrentIndex(0)
        layout.addWidget(panel.symbol, 0, 0)
        # panel.symbol.currentTextChanged.connect(self.load_asset)

        layout.setColumnMinimumWidth(1, 30)

        panel.period = QComboBox(panel)
        [panel.period.addItem(i) for i in '1m'.split()]
        panel.period.setCurrentIndex(0)
        layout.addWidget(panel.period, 0, 2)
        # panel.period.currentTextChanged.connect(self.load_asset)

        layout.setColumnMinimumWidth(3, 30)

        panel.indicators = QComboBox(panel)
        [panel.indicators.addItem(i) for i in 'Clean:Few indicators:Moar indicators'.split(':')]
        panel.indicators.setCurrentIndex(0)
        layout.addWidget(panel.indicators, 0, 4)
        # panel.indicators.currentTextChanged.connect(self.load_asset)

        layout.setColumnMinimumWidth(5, 30)

        panel.history = QLineEdit(panel)
        panel.history.setValidator(QtGui.QIntValidator(1080, 10080))
        panel.history.setPlaceholderText("  History")
        layout.addWidget(panel.history, 1, 0)

        layout.setColumnMinimumWidth(7, 30)

        panel.tolerance = QLineEdit(panel)
        panel.tolerance.setValidator(QtGui.QDoubleValidator(0.001, 1, 3))
        panel.tolerance.setPlaceholderText("  Tolerance")
        layout.addWidget(panel.tolerance, 1, 2)

        layout.setColumnMinimumWidth(9, 30)

        panel.tolerance2 = QLineEdit(panel)
        panel.tolerance2.setValidator(QtGui.QDoubleValidator(0.000001, 1000000, 6))
        panel.tolerance2.setPlaceholderText("  Tolerance 2")
        layout.addWidget(panel.tolerance2, 1, 4)

        layout.setColumnMinimumWidth(11, 30)

        panel.method = QComboBox(panel)
        [panel.method.addItem(i) for i in 'Method_1 Method_2 Method_3'.split()]
        panel.method.setCurrentIndex(1)
        layout.addWidget(panel.method, 2, 0)

        layout.setColumnMinimumWidth(13, 30)

        panel.filter = QComboBox(panel)
        [panel.filter.addItem(i) for i in 'All Medium High'.split()]
        panel.filter.setCurrentIndex(0)
        layout.addWidget(panel.filter, 2, 2)

        panel.grab = QPushButton('Save screen')
        panel.grab.clicked.connect(grab_screenshot)
        layout.addWidget(panel.grab, 2, 4)

        panel.refresh = QPushButton('Compute')
        panel.refresh.clicked.connect(self.launcher)
        layout.addWidget(panel.refresh, 3, 0)

        layout.setColumnMinimumWidth(15, 30)

        panel.progressBar = QProgressBar(panel)
        panel.progressBar.setRange(0, 100)
        panel.progressBar.setValue(0)
        layout.addWidget(panel.progressBar, 3, 2)

        layout.setColumnMinimumWidth(17, 30)

        panel.time = QLabel(panel)
        panel.time.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(panel.time, 3, 4)

        return panel

    def launcher(self):
        fplt._clear_timers()
        method = ctrl_panel.method.currentText().lower()
        period = ctrl_panel.period.currentText()
        tol = float(ctrl_panel.tolerance.text())
        dummy = None
        if method != 'method_3':
            self.df, self.df_time = lp.load_asset(period)
            cr_star, df, ts_pivots = lp.targets(self.df, self.df_time, 5, 5, period, tol)
            lp.plotting(cr_star, self.df_time, method)
            fplt.timer_callback(self.realtime_update_plot, 5)  # update every 5 seconds
            fplt.refresh()
        if method == 'method_3':
            self.df, self.df_time = lp.load_asset('1h')
            df_h1, dummy1, dummy2 = lp.targets(self.df, self.df_time, 5, 5, '1h', 25)  # XBT: 25 / ETH: 2.5

            logger.info('Tier 1/3 done...')
            self.df, self.df_time = lp.load_asset('5m')
            df_m5, dummy1, dummy2 = lp.targets(self.df, self.df_time, 5, 5, '5m', 3.5)  # XBT: 3.5 / ETH: 0.35

            logger.info('Tier 2/3 done...')
            self.df, self.df_time = lp.load_asset('1m')
            df_m1, dummy1, dummy2 = lp.targets(self.df, self.df_time, 5, 5, '1m', 1)  # XBT: 1 / ETH: 0.5

            logger.info('Tier 3/3 done... Plotting !')
            lp.plotting(df_m1, self.df_time, method)

            start_date = pd.to_datetime(df_m1['timing']).iloc[0]
            end_date = start_date + timedelta(minutes=(self.history + 1440))
            logger.info("Beginning = " + str(start_date) + " / End = " + str(end_date))

            i = start_date
            count = 0
            offset = 0
            offset_2 = 0
            accelerator = 0

            for h in range(len(df_h1)):
                if pd.to_datetime(df_h1['timing'].iloc[h], utc=True) > start_date:
                    # print(pd.to_datetime(df_h1['timing'].iloc[h], utc=True))
                    offset = h
                    break

            for g in range(len(df_m5)):
                if pd.to_datetime(df_m5['timing'].iloc[g], utc=True) >= pd.to_datetime(
                        df_h1['timing'].iloc[offset] - timedelta(minutes=55), utc=True):
                    # print(pd.to_datetime(df_m5['timing'].iloc[g], utc=True))
                    offset_2 = g
                    break

            # print('offsets: ', offset, ' / ', offset_2)

            for j in range(offset, len(df_h1), 1):
                if isinstance(df_h1.value.iloc[j], list) is True:
                    # print("Ping_1: ", str(df_h1.iloc[j]))
                    start_date_2 = pd.to_datetime(df_h1['timing'].iloc[j], utc=True) - timedelta(minutes=60)
                    end_date_2 = pd.to_datetime(df_h1['timing'].iloc[j], utc=True)
                    k = start_date_2
                    for m in range(offset_2, len(df_m5), 1):
                        if pd.to_datetime(df_m5['timing'].iloc[m], utc=True) >= start_date_2 and isinstance(
                                df_m5.value.iloc[m], list) is True:
                            # print("Ping_2: ", str(df_m5.iloc[m]))
                            start_date_3 = pd.to_datetime(df_m5['timing'].iloc[m], utc=True) - timedelta(minutes=5)
                            end_date_3 = pd.to_datetime(df_m5['timing'].iloc[m], utc=True)
                            n = start_date_3
                            for p in range(accelerator, len(df_m1), 1):
                                # accelerator += 1
                                if pd.to_datetime(df_m1['timing'].iloc[p], utc=True) >= start_date_3:
                                    n += timedelta(minutes=1)
                                    if n > end_date_3:
                                        break
                                    if isinstance(df_m1.value.iloc[p], list) is True:
                                        print("Bingo: " + str(pd.to_datetime(df_m1.timing.iloc[p])))
                                        self.draw(dummy, self.df_time, 'v', df_m1['timing'][p], ax=ax, color='#ff7f00')
                                        count += 1
                        if pd.to_datetime(df_m5['timing'].iloc[m], utc=True) >= start_date_2:
                            k += timedelta(minutes=5)
                        if k > end_date_2:
                            break
                i += timedelta(minutes=60)
                if i > end_date:
                    # print('i: ', i)
                    break

            logger.info('End of Analysis: ' + str(count) + " MTF alignment(s) spotted !")
            fplt.timer_callback(self.realtime_update_plot, 5)  # update every 5 seconds
            fplt.refresh()


    # @njit(nogil=True)
    def getExtrapoledLine(self, p1a, p1b, p2a, p2b):
        '''Creates a line extrapoled in p1->p2 direction'''
        EXTRAPOL_RATIO = 1000
        a = p1a, p1b
        b = (p1a + EXTRAPOL_RATIO * (p2a - p1a), p1b + EXTRAPOL_RATIO * (p2b - p1b))
        return a, b

    # @njit(nogil=True)  #, parallel=True)
    def slope_from_points(self, point1a, point1b, point2a, point2b):
        slope_ = (point2b - point1b) / (point2a - point1a)
        return slope_

    def AngleBtw2Points(self, pointA, pointB):
        changeInX = pointB[0] - pointA[0]
        changeInY = pointB[1] - pointA[1]
        return round(degrees(atan2(changeInY, changeInX)), 3)

    def targets(self, df_, df_time, lookahead, delta, period, tol):
        try:
            dummy = []
            df = df_[:-offset_graph]
            df.index = df_time.Time[:-offset_graph]

            df_high = df.drop(labels=['Time', 'Open', 'Low', 'Close', 'Volume'], axis=1)
            df_high.index = df.index
            df_high = df_high.to_numpy()

            df_low = df.drop(labels=['Time', 'Open', 'High', 'Close', 'Volume'], axis=1)
            df_low.index = df.index
            df_low = df_low.to_numpy()

            df_ind = df.copy()
            del df_ind['Time']
            del df_ind['Volume']

            pivots_h = peakdetect.peakdetect(df_high, lookahead=lookahead, delta=delta)
            pivots_l = peakdetect.peakdetect(-df_low, lookahead=lookahead, delta=delta)

            df_pv = pd.DataFrame(columns=['High', 'Low', 'Time', 'Price'])

            ts_pivots_h = pd.Series(pivots_h[0])
            ts_pivots_l = pd.Series(pivots_l[0])

            for i in ts_pivots_h:
                df_pv.at[i[0], 'High'] = i[1]
                df_pv.at[i[0], 'Time'] = i[0]

            for i in ts_pivots_l:
                df_pv.at[i[0], 'Low'] = i[1]
                df_pv.at[i[0], 'Time'] = i[0]

            df_pv.set_index('Time', drop=True, inplace=True)
            df_pv.sort_index(ascending=True, inplace=True)
            df_pv.drop('Price', axis=1, inplace=True)
            df_pv.fillna(0, inplace=True)
            df_pv.High = df_pv.High.astype(float)
            df_pv.Low = df_pv.Low.astype(float)
            for i in range(0, len(df_pv)):
                if df_pv.High.iloc[i] != 0 and df_pv.Low.iloc[i] != 0:
                    df_pv.High.iloc[i] = df_pv.Low.iloc[i] = 0
            df_pv = df_pv.High + df_pv.Low
            df_pv = df_pv.reindex(range(0, self.history), fill_value=0)
            df_pv = pd.DataFrame(data=df_pv, columns=['Price']).astype({"Price": 'float32'})
            ts_pivots = pd.DataFrame(columns=['Time', 'Price'])
            ts_pivots.Time = mpl_dates.num2epoch(mpl_dates.date2num(df_ind.index))
            ts_pivots.Price = df_pv.Price
            ts_pivots.query("Price != 0", inplace=True)
            ts_pivots.Price = abs(ts_pivots.Price)
            ts_pivots.reset_index(drop=True, inplace=True)
            # ts_pivots.to_csv('results_' + str(symbol) + '_' + str(period) + '_' + str(exchange) + '_tspivot.csv')

            k = -1
            buff = pd.DataFrame(columns=['Time', 'Price'])

            lines_matrix = pd.DataFrame(columns=['Lines']) #.astype({"Lines": 'complex'})
            tempo = 0

            for i in range(len(ts_pivots)):
                if ts_pivots['Price'][i] == 0:
                    continue
                else:
                    j = i
                    p1 = ts_pivots.at[j, 'Price']
                    t1 = ts_pivots.at[j, 'Time']
                    while j < len(ts_pivots) - 5:
                        j += 1
                        if ts_pivots['Price'][j + 2] != 0 and ts_pivots['Price'][j + 4] != 0:
                            k += 1
                            p2 = ts_pivots['Price'][j + 2]
                            t2 = ts_pivots.Time[j + 2]
                            p3 = ts_pivots['Price'][j + 4]
                            t3 = ts_pivots.Time[j + 4]
                            buff.at[k, 'Price'] = p2
                            buff.at[k, 'Time'] = t2
                            proxy1 = t1
                            proxy2 = t2
                            lines_matrix.at[tempo, 'Lines'] = [[proxy1, p1], [proxy2, p2]]
                            tempo += 1
                            if k > 1 and i > 1:
                                proxy1 = buff['Time'].loc[k - 2]
                                proxy2 = t2
                                lines_matrix.at[tempo, 'Lines'] = [[proxy1, buff['Price'].loc[k - 2]], [proxy2, p2]]
                                tempo += 1
                                #####################
                                if (p1 < p2 and p3 < p2 and p3 > p1) or (p1 > p2 and p3 > p2 and p1 > p3):
                                    proxy1 = t2
                                    proxy2 = t3
                                    lines_matrix.at[tempo, 'Lines'] = [[proxy1, p2], [proxy2, p3]]
                                    tempo += 1
                            break
                    continue

            cr_coo = pd.DataFrame(index=range(1000000), columns=['cross_timing', 'cross_price'])

            o = 0
            maxVal = len(lines_matrix) - 1

            logger.info('Detection of Crosses on ' + Fore.LIGHTCYAN_EX + str(period.upper()))
            for m in range(0, len(lines_matrix), 1):

                if ctrl_panel is not None:
                    curVal = m
                    to_display = round((curVal / maxVal) * 100, 0)
                    ctrl_panel.progressBar.setValue(int(to_display))
                    QApplication.processEvents()

                T1A = lines_matrix.Lines.iloc[m][0]
                p_T1A = lines_matrix.Lines.iloc[m][0][1]
                t_T1A = lines_matrix.Lines.iloc[m][0][0]
                T2A = lines_matrix.Lines.iloc[m][1]
                p_T2A = lines_matrix.Lines.iloc[m][1][1]
                t_T2A = lines_matrix.Lines.iloc[m][1][0]

                for n in range(m + 1, len(lines_matrix), 1):
                    if m == n:
                        continue

                    T1B = lines_matrix.Lines.iloc[n][0]
                    p_T1B = lines_matrix.Lines.iloc[n][0][1]
                    t_T1B = lines_matrix.Lines.iloc[n][0][0]
                    T2B = lines_matrix.Lines.iloc[n][1]
                    p_T2B = lines_matrix.Lines.iloc[n][1][1]
                    t_T2B = lines_matrix.Lines.iloc[n][1][0]

                    if p_T1A == p_T1B or p_T2A == p_T2B:
                        continue

                    coo1a, coo1b = self.getExtrapoledLine(T1A[0], T1A[1], T2A[0], T2A[1])
                    line1a = LineString([coo1a, coo1b])
                    coo2a, coo2b = self.getExtrapoledLine(T1B[0], T1B[1], T2B[0], T2B[1])
                    line2a = LineString([coo2a, coo2b])

                    cr_coo_1 = line1a.intersection(line2a)
                    cr_coo_1 = list(cr_coo_1.coords)

                    if len(cr_coo_1) != 0 and p_T1A != cr_coo_1[0][1] and p_T2A != cr_coo_1[0][1] and p_T1B != cr_coo_1[0][1] and p_T2B != cr_coo_1[0][1] \
                            and t_T1A != cr_coo_1[0][0] and t_T2A != cr_coo_1[0][0] and t_T1B != cr_coo_1[0][0] and t_T2B != cr_coo_1[0][0]:
                        cr_coo.at[o, 'cross_timing'] = cr_coo_1[0][0]
                        cr_coo.at[o, 'cross_price'] = round(cr_coo_1[0][1], 2)
                        o += 1

            logger.info('Number of Crosses: ' + Fore.LIGHTCYAN_EX + str(o))

            cr_coo.dropna(inplace=True)
            # cr_coo.to_csv('results_' + str(symbol) + '_' + str(period) + '_' + str(exchange) + '_crcoo.csv')

            cr_star = cr_coo.copy()
            cr_star = cr_star.sort_values(by='cross_timing', axis=0, ascending=True)
            cr_star.cross_timing = mpl_dates.num2date(mpl_dates.epoch2num(cr_star.cross_timing))
            cr_star = cr_star.set_index(['cross_timing'])

            def custom_resampler(array):
                list = []
                g = 0
                cum_prices = 0
                if len(array) == 0:
                    return None
                for i in range(len(array)):
                    list.append(array[i])
                if len(list) <= 3:
                    return None
                list = sorted(list)
                for j in range(len(list)-1):
                    if abs(list[j] - list[j+1]) <= tol and list[j] > 0 and list[j+1] > 0:  # Define a dollar tolerance
                        if g == 0:
                            cum_prices += (list[j] + list[j+1])
                        else:
                            cum_prices += list[j+1]
                        g += 1
                        continue
                    if 4 > g > 0 and abs(list[j] - list[j+1]) > tol:
                        g = 0
                        cum_prices = 0
                    if g >= 4 and abs(list[j] - list[j+1]) > tol:
                        break
                if g <= 3:
                    return None
                else:
                    return [round(cum_prices / (g + 1), 3), g]

            if period == '1m':
                cr_star = cr_star.resample('1T', label='right', closed='right').apply(custom_resampler)
            if period == '5m':
                cr_star = cr_star.resample('5T', label='right', closed='right').apply(custom_resampler)
            if period == '1h':
                cr_star = cr_star.resample('60T', label='right', closed='right').apply(custom_resampler)
            if period == '1d':
                cr_star = cr_star.resample('1440T', label='right', closed='right').apply(custom_resampler)

            cr_star = cr_star.reset_index()

            # cr_star.query("cross_price != 0", inplace=True)
            cr_star.columns = ['timing', 'value']
            cr_star.to_csv('results_' + str(ctrl_panel.symbol.currentText()) + '_' + str(period) + '_star.csv')
            return cr_star, df, ts_pivots
        except KeyboardInterrupt:
            sys.exit()

    def plotting(self, cr_star, df_time, method):
        o = 0
        dummy = None
        triple_cross = []
        tol2 = float(ctrl_panel.tolerance2.text())
        filtration = ctrl_panel.filter.currentText().lower()

        # for i in range(len(ts_pivots)):
        #    fplt.plot(ts_pivots.Time[i], ts_pivots.Price[i], ax=ax, color='#4a5', style='o')

        for i in range(len(cr_star)):
            if cr_star.value[i] is None:
                continue
            else:
                j = self.get_index_reference(df_time, cr_star['timing'][i])
                if j is None:
                    continue
                # fplt.plot(cr_star['timing'][i], cr_star['value'][i][0], ax=ax, color='#4b8', style='x')
                if cr_star['value'][i][1] <= 5 and filtration == 'all':
                    self.draw(dummy, df_time, 'v', cr_star['timing'][i], ax=ax, color='#FFC0CB')
                if 9 > cr_star['value'][i][1] > 5 and (filtration == 'all' or filtration == 'medium'):
                    self.draw(dummy, df_time, 'v', cr_star['timing'][i], ax=ax, color='#52a6bd')
                if cr_star['value'][i][1] >= 9 and (filtration == 'all' or filtration == 'medium' or filtration == 'high'):
                    self.draw(dummy, df_time, 'v', cr_star['timing'][i], ax=ax, color='#a0ee45')
                triple_cross.append([i, cr_star['timing'][i], j, cr_star['value'][i][0], cr_star['value'][i][1]])
                o += 1

        logger.info('Number of Triple Crosses: ' + Fore.LIGHTCYAN_EX + str(o))

        m = 0
        r = 0
        maxVal = len(triple_cross) - 2
        color_pool = ['magenta', 'blue', 'red', 'black', 'green']

        for i in range(len(triple_cross) - 1):

            if ctrl_panel is not None:
                curVal = i
                to_display = round((curVal / maxVal) * 100, 0)
                ctrl_panel.progressBar.setValue(int(to_display))
                QApplication.processEvents()

            ti_1a = triple_cross[i][2]
            pr_1a = triple_cross[i][3]
            ti_2a = triple_cross[i + 1][2]
            pr_2a = triple_cross[i + 1][3]
            # slope_1 = slope_from_points(ti_1a, pr_1a, ti_1b, pr_1b)
            angle_1 = lp.AngleBtw2Points([ti_1a, pr_1a], [ti_2a, pr_2a])
            if -80 > angle_1 or angle_1 > 80:
                continue
            done = False
            for j in range(i + 2, len(triple_cross), 1):
                ti_3a = triple_cross[j][2]
                pr_3a = triple_cross[j][3]
                if method == 'method_1':
                    for k in range(j + 1, len(triple_cross), 1):
                        ti_4a = triple_cross[k][2]
                        pr_4a = triple_cross[k][3]
                        # slope_2 = slope_from_points(ti_2a, pr_2a, ti_2b, pr_2b)
                        angle_2 = lp.AngleBtw2Points([ti_3a, pr_3a], [ti_4a, pr_4a])
                        if -80 > angle_2 or angle_2 > 80:
                            continue
                        # diff = (slope_2/(slope_1 * 1.0))
                        if angle_1 == 0:
                            diff = 0
                        else:
                            diff = (angle_2 / (angle_1 * 1.0))
                        if 1 - tol2 < diff < 1 + tol2:
                            color = color_pool[m]
                            m += 1
                            if m > 4:
                                m = 0
                            if done is False:
                                a, b = [ti_1a, triple_cross[i][3]], [ti_2a, triple_cross[i + 1][3]]
                                line = LineString([a, b])
                                lineList = list(line.coords)
                                container1 = a
                                angle = lp.AngleBtw2Points(lineList[0], lineList[1])
                                force_line = pg.InfiniteLine(pos=container1, angle=angle, movable=False)
                                force_line.setPen(color='#4b8', style=QtCore.Qt.SolidLine)
                                ax.addItem(force_line, ignoreBounds=True)
                                r += 1
                                done = True
                            c, d = [ti_3a, triple_cross[j][3]], [ti_4a, triple_cross[k][3]]
                            line = LineString([c, d])
                            lineList = list(line.coords)
                            container2 = c
                            angle = lp.AngleBtw2Points(lineList[0], lineList[1])
                            force_line = pg.InfiniteLine(pos=container2, angle=angle, movable=False)
                            force_line.setPen(color='#4b8', style=QtCore.Qt.SolidLine)
                            ax.addItem(force_line, ignoreBounds=True)
                            r += 1
                            break
                if method == 'method_2' or method == 'method_3':
                    for k in range(j + 1, len(triple_cross), 1):
                        ti_1a = triple_cross[i][2]
                        pr_1a = triple_cross[i][3]
                        ti_2a = triple_cross[j][2]
                        pr_2a = triple_cross[j][3]
                        ti_3a = triple_cross[k][2]
                        pr_3a = triple_cross[k][3]
                        area = (ti_1a * (pr_2a - pr_3a) + ti_2a * (pr_3a - pr_1a) + ti_3a * (pr_1a - pr_2a))
                        if tol2 >= area >= -tol2:
                            color_code = 1

                            for l in range(k + 1, len(triple_cross), 1):
                                ti_4a = triple_cross[l][2]
                                pr_4a = triple_cross[l][3]
                                area = (ti_1a * (pr_3a - pr_4a) + ti_3a * (pr_4a - pr_1a) + ti_4a * (pr_1a - pr_3a))
                                if tol2 >= area >= -tol2:
                                    color_code += 1

                            a, c = [ti_1a, pr_1a], [ti_3a, pr_3a]
                            line = LineString([a, c])
                            lineList = list(line.coords)
                            container1 = a
                            angle = lp.AngleBtw2Points(lineList[0], lineList[1])
                            force_line = pg.InfiniteLine(pos=container1, angle=angle, movable=False)
                            if color_code <= 2 and filtration == 'all':
                                force_line.setPen(color='#FFC0CB', style=QtCore.Qt.SolidLine)
                            if 2 < color_code <= 5 and (filtration == 'all' or filtration == 'medium'):
                                force_line.setPen(color='#52a6bd', style=QtCore.Qt.SolidLine)
                            if 5 < color_code and (filtration == 'all' or filtration == 'medium' or filtration == 'high'):
                                force_line.setPen(color='#a0ee45', style=QtCore.Qt.SolidLine)
                            ax.addItem(force_line, ignoreBounds=True)
                            r += 1
                            break

        if method == str('method_1'):
            logger.info('Number of //: ' + Fore.LIGHTCYAN_EX + str(r))
        if method == str('method_2') or method == str('method_3'):
            logger.info('Number of --: ' + Fore.LIGHTCYAN_EX + str(r))

############################################################################
############################################################################

if __name__ == '__main__':
    # use websocket for real-time
    # ws = BitMEXFutureWebsocket(symbol=symbol, history=history, api_key='', api_secret='')
    ax, ax_vol, ax_rsi = fplt.create_plot('GodRays Analyzer v1.0 [BitMEX Futures]', rows=3, init_zoom_periods=300)

    fplt.display_timezone = timezone.utc

    plots = {}
    fplt.y_pad = 0.07 # pad some extra (for control panel)
    fplt.max_zoom_points = 7
    fplt.autoviewrestore()
    axo = ax_vol

    # hide rsi chart to begin with; show x-axis of top plot
    ax_rsi.hide()
    ax_rsi.vb.setBackgroundColor(None) # don't use odd background color
    ax.set_visible(xaxis=True)

    try:
        lp = Live_Analysis()
        ctrl_panel = lp.create_ctrl_panel(ax.vb.win)
        fplt.refresh()
        fplt.show()
    except KeyboardInterrupt:
        sys.exit()
