import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

pd.options.mode.chained_assignment = None  # default='warn'

# -----------------------------------------Example
# df = pd.read_csv("xauusd.csv", header=0,
#                  usecols=["close", 'high', 'low', 'time', 'open', 'time'])
# df = df[-1000:].reset_index()
# model = {
#     "pivot_connect1": 2,
#     "counter_limit1": 100,
#     "degree_limit1": 1,
#     "distance_limit1": 10,
#     "countinues_limit1": 3,
#     "swing_percentage1": 0.03,
#     "degree_side_limit1": "up",
#     "band_perc1": 0.007,
#     "band_increase_step1": 0.000025,
# }


def draw_trendline(model, ClosePrice, HighPrice, LowPrice, OpenPrice, Index,plot:bool=True):
    """

    :param model: Send Options Like Example Model (dont change name of parametrs)
    :param ClosePrice: Send Close,High,Low and Open Price As a Pandas DataFrame
    :param Index: Send Index of DataFrame (if u want see gaps send time!
    :param plot: If u want plot trend lines, pivots and ohlc with plotly set True
    :return: TrendLineBullish, TrendLineBearish, High_Pivots, Low_Pivots
    TrendLineBullish = [xlbull, ylbull, upband_bull, lowband_bull, start_bull]
    TrendLineBearish = [xlbear, ylbear, upband_bear, lowband_bear, start_bear]
    High_Pivots = [peak_x, peak_y,peak_x_real]
    Low_Pivots = [trough_x, trough_y, trough_x_real]

    on xlbull or xlbear and ybull and upperbands each index represent of a trend line
    if u dont know to use it see plot section to see how i use it
    """
    def Swings(ClosePrice, HighPrice, LowPrice, perc):
        """
        :param closep: pass the Close price Series
        :param HighPrice: pass the High price series
        :param low: pass the Low price series
        :param perc: pass the percentage of Swings
        :return: 1: high pivot index 2: high pivot price 3: low pivot index 4: low pivot value 5. high real index
        6. low real index  (real index is where pivot determinated and so important in back testing)
        """
        try:
            chk = {
                'trend': '', 'sec_chk': '', 'check_reserve': '', 'res_low': 0, 'res_high': 0,
                'low_index': 0, 'high_index': 0, 'high': ClosePrice[0], 'low': ClosePrice[0]}
            pvt_high = []
            pvt_low = []
            real_i_low = []
            real_i_high = []
            for i in range(1, len(ClosePrice)):
                if chk['trend'] == '':
                    if (HighPrice[i] - chk['low']) / chk['low'] * 100 > perc:
                        # pvt_low.append(i)
                        chk['high'] = HighPrice[i]
                        chk['res_high'] = i
                        chk['trend'] = 'high'
                    elif (chk['high'] - LowPrice[i]) / chk['high'] * 100 >= perc:
                        # pvt_high.append(i)
                        chk['low'] = LowPrice[i]
                        chk['trend'] = 'low'
                        chk['res_low'] = i
                if chk['trend'] == 'high':
                    if HighPrice[i] > chk['high']:
                        chk['high'] = HighPrice[i]
                        chk['res_high'] = i
                    elif (chk['high'] - LowPrice[i]) / chk['high'] * 100 >= perc:
                        pvt_high.append(chk['res_high'])
                        real_i_high.append(i)
                        chk['low'] = LowPrice[i]
                        chk['trend'] = 'low'
                        chk['res_low'] = i
                elif chk['trend'] == 'low':
                    if LowPrice[i] < chk['low']:
                        chk['low'] = LowPrice[i]
                        chk['res_low'] = i
                    elif (HighPrice[i] - chk['low']) / chk['low'] * 100 > perc:
                        pvt_low.append(chk['res_low'])
                        real_i_low.append(i)
                        chk['high'] = HighPrice[i]
                        chk['trend'] = 'high'
                        chk['res_high'] = i
            pvt = []
            pvth = []
            pvtl = []
            cc = 0
            for i in range(0, len(ClosePrice)):
                checked = False
                chkh = False
                chkl = False
                for s in range(0, len(pvt_high)):
                    if pvt_high[s] == i:
                        pvt.append(HighPrice[i])
                        chkh = True
                        pvth.append(HighPrice[i])
                        cc += 1
                        checked = True
                        break
                if not checked:
                    for s in range(0, len(pvt_low)):
                        if pvt_low[s] == i:
                            pvt.append(LowPrice[i])
                            chkl = True
                            pvtl.append(LowPrice[i])
                            cc += 1
                            checked = True
                            break
                if not chkl:
                    pvtl.append(np.NaN)
                if not chkh:
                    pvth.append(np.NaN)
                if not checked:
                    pvt.append(np.NaN)
            high_index = []
            high_value = []
            high_index_real = []
            low_index = []
            low_value = []
            low_index_real = []
            for i in range(0, len(pvt_high)):
                high_index.append(pvt_high[i])
                high_value.append(HighPrice.iloc[pvt_high[i]])
                high_index_real.append(real_i_high[i])
            for i in range(0, len(pvt_low)):
                low_index.append(pvt_low[i])
                low_value.append(LowPrice.iloc[pvt_low[i]])
                low_index_real.append(real_i_low[i])
            df87678 = pd.DataFrame()
            df87678['high_index'] = high_index
            df87678['high_value'] = high_value
            df87678['high_index_real'] = high_index_real
            df85965 = pd.DataFrame()
            df85965['low_index'] = low_index
            df85965['low_value'] = low_value
            df85965['low_index_real'] = low_index_real
            return df87678['high_index'], df87678['high_value'], df85965['low_index'], df85965['low_value'], \
                   df87678['high_index_real'], df85965['low_index_real']
        except Exception as error:
            print("SomeThing wrong in Swings : ",error)
    def trendline_bearish(peak_x, peak_y, peak_x_real, closeprice, pivot_connect: int = 2, counter_limit: int = 4,
                          degree_limit: float = 2.5, distance_limit: int = 10, countinues_limit: int = 10,
                          degree_side_limit: str = "up", band_perc: float = 0.01,
                          band_increase_step: float = 0.001):
        """

        :param peak_x: pass the high pivots u got from zigzag
        :param peak_y: pass the value of high pivots from zigzag
        :param peak_x_real: pass the real index of high pivots
        :param closeprice: pass the close price as pandas DataFrame
        :param pivot_connect: How much pivot connect together to draw trend Line?
        :param counter_limit: how u much pivot failed after first pivot?
        :param degree_limit: Trend Line Degree Limit (dont forget to set the side of degree limit)
        :param distance_limit: distance from first pivot and last pivot of trend line
        :param countinues_limit: after draw trend line set the limit for delete smalls one that break after 1 candle
        :param degree_side_limit: if its set to "dn" this says lower the degree limit and "up" is draw upper than that degree
        :param band_perc: drawing a band upper and lower of Trend Line
        :param band_increase_step: if u want after draw trend line band increasing set the number
        :return: valid_trend_line_x, valid_trend_line_y, upperband, lowerband, start(start index of trend line !important for backtesting)
        """
        try:
            win = pivot_connect
            buff_x = [peak_x[0]]
            buff_y = [peak_y.iloc[0]]
            buff_x_real = [peak_x_real.iloc[0]]
            trend_x, trend_x_real, trend_y = [], [], []
            counter = 0
            for i in range(0, len(peak_x)):
                if len(buff_x) >= win:
                    trend_x.append(buff_x)
                    trend_y.append(buff_y)
                    trend_x_real.append(buff_x_real)
                    buff_x = [peak_x[i - 1]]
                    buff_y = [peak_y.iloc[i - 1]]
                    buff_x_real = [peak_x_real.iloc[i - 1]]
                    counter = 0
                if len(buff_x) <= win and peak_y.iloc[i] < buff_y[-1]:
                    buff_x.append(peak_x[i])
                    buff_y.append(peak_y.iloc[i])
                    buff_x_real.append(peak_x_real.iloc[i])
                    counter = 0
                else:
                    counter += 1
                if peak_y.iloc[i] > buff_y[-1]:
                    buff_x = [peak_x[i]]
                    buff_y = [peak_y.iloc[i]]
                    buff_x_real = [peak_x_real.iloc[i]]
                    counter = 0

                if counter >= counter_limit:
                    buff_x = [peak_x[i]]
                    buff_y = [peak_y.iloc[i]]
                    buff_x_real = [peak_x_real.iloc[i]]

            last_x = []
            last_y = []
            first_x = []
            first_y = []
            steps = []
            last_x_real = []
            for i in trend_x:
                last_y.append(0)
                last_x.append(0)
                first_y.append(0)
                first_x.append(0)
                steps.append(0)
                last_x_real.append(0)
            for i in range(0, len(trend_x)):
                first_y[i] = trend_y[i][0]
                first_x[i] = trend_x[i][0]
                last_x[i] = trend_x[i][-1]
                last_y[i] = trend_y[i][-1]
                last_x_real[i] = trend_x_real[i][-1]
                steps[i] = (first_y[i] - last_y[i]) / (last_x[i] - first_x[i])
            trend_line_x = []
            trend_line_y = []
            trend_line_x_real = []
            for i in range(0, len(trend_x)):
                p1 = np.array([first_x[i], first_y[i]])
                p2 = np.array([last_x[i], last_y[i]])
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                angle_radians = np.arctan(slope)
                angle_degrees = np.degrees(angle_radians)
                if degree_side_limit == "up":
                    if abs(angle_degrees) >= degree_limit:
                        trend_line_x.append([first_x[i]])
                        trend_line_y.append([first_y[i]])
                        trend_line_x_real.append([last_x_real[i]])
                        for j in range(first_x[i] + 1, last_x[i] + 1):
                            trend_line_x[-1].append(j)
                            trend_line_y[-1].append(trend_line_y[-1][-1] - steps[i])
                elif degree_side_limit == "dn":
                    if abs(angle_degrees) <= degree_limit:
                        trend_line_x.append([first_x[i]])
                        trend_line_y.append([first_y[i]])
                        trend_line_x_real.append([last_x_real[i]])
                        for j in range(first_x[i] + 1, last_x[i] + 1):
                            trend_line_x[-1].append(j)
                            trend_line_y[-1].append(trend_line_y[-1][-1] - steps[i])
            valid_trend_line_x = []
            valid_trend_line_y = []
            start = []
            band_arr = []
            for i in range(0, len(trend_line_y)):
                try:
                    valid = True
                    if trend_line_x[i][-1] - trend_line_x[i][0] < distance_limit:
                        valid = False
                    else:
                        for j in range(1, len(trend_line_y[i]) - 2):
                            if closeprice.iloc[trend_line_x[i][j]] > trend_line_y[i][j]:
                                valid = False
                                break
                    if valid:
                        step1 = trend_line_y[i][0] - trend_line_y[i][1]
                        band_buff = band_perc
                        count = 0
                        start_index = trend_line_x_real[i][0]
                        for l in range(trend_line_x[i][-1] + 1, trend_line_x[i][-1] + 5000000):
                            if l + 1 < len(closeprice):
                                if closeprice.iloc[l + 1] <= trend_line_y[i][-1] - step1 + (
                                        (trend_line_y[i][-1] - step1) * (band_buff / 100)):
                                    trend_line_y[i].append(trend_line_y[i][-1] - step1)
                                    trend_line_x[i].append(l)
                                    count += 1
                                    band_buff += band_increase_step
                                else:
                                    trend_line_y[i].append(trend_line_y[i][-1] - step1)
                                    trend_line_x[i].append(l)
                                    trend_line_y[i].append(trend_line_y[i][-1] - step1)
                                    trend_line_x[i].append(l + 1)
                                    count += 1
                                    break
                            else:
                                trend_line_y[i].append(trend_line_y[i][-1] - step1)
                                trend_line_x[i].append(l)
                                count += 1
                                break
                        if count >= countinues_limit:
                            valid_trend_line_x.append(trend_line_x[i])
                            valid_trend_line_y.append(trend_line_y[i])
                            band_arr.append(band_buff)
                            start.append(start_index)
                except:
                    print("error")

            upperband = []
            lowerband = []
            for i in range(0, len(valid_trend_line_y)):
                arrup = []
                arrdn = []
                band_buff = band_perc
                for j in range(0, len(valid_trend_line_y[i])):
                    arrup.append(valid_trend_line_y[i][j] + (valid_trend_line_y[i][j] * (band_buff / 100)))
                    arrdn.append(valid_trend_line_y[i][j] - (valid_trend_line_y[i][j] * (band_buff / 100)))
                    band_buff += band_increase_step
                upperband.append(arrup)
                lowerband.append(arrdn)
            return valid_trend_line_x, valid_trend_line_y, upperband, lowerband, start
        except Exception as er:
            print("Something Wrong in Bearish Trend Line : ",er)

    def trendline_bullish(trough_x, trough_y, trough_x_real, closeprice, pivot_connect: int = 2, counter_limit: int = 4,
                          degree_limit: float = 2.5, distance_limit: int = 10, countinues_limit: int = 10,
                          degree_side_limit: str = "up", band_perc: float = 0.01,
                          band_increase_step: float = 0.001):
        """

        :param trough_x: pass the low pivots u got from zigzag
        :param trough_y: pass the value of low pivots from zigzag
        :param trough_x_real: pass the real index of low pivots
        :param closeprice: pass the close price as pandas DataFrame
        :param pivot_connect: How much pivot connect together to draw trend Line?
        :param counter_limit: how u much pivot failed after first pivot?
        :param degree_limit: Trend Line Degree Limit (dont forget to set the side of degree limit)
        :param distance_limit: distance from first pivot and last pivot of trend line
        :param countinues_limit: after draw trend line set the limit for delete smalls one that break after 1 candle
        :param degree_side_limit: if its set to "dn" this says lower the degree limit and "up" is draw upper than that
         degree
        :param band_perc: drawing a band upper and lower of Trend Line
        :param band_increase_step: if u want after draw trend line band increasing set the number
        :return: valid_trend_line_x, valid_trend_line_y, upperband, lowerband, start(start index of trend line
        !important for backtesting)
        """
        try:
            win = pivot_connect
            buff_x = [trough_x[0]]
            buff_y = [trough_y.iloc[0]]
            buff_x_real = [trough_x_real.iloc[0]]
            trend_x = []
            trend_x_real = []
            trend_y = []
            counter = 0
            for i in range(0, len(trough_x)):
                if len(buff_x) >= win:
                    trend_x.append(buff_x)
                    trend_y.append(buff_y)
                    trend_x_real.append(buff_x_real)
                    buff_x = [trough_x[i - 1]]
                    buff_y = [trough_y.iloc[i - 1]]
                    buff_x_real = [trough_x_real.iloc[i - 1]]
                    counter = 0
                if len(buff_x) <= win and trough_y.iloc[i] > buff_y[-1]:
                    buff_x.append(trough_x[i])
                    buff_y.append(trough_y.iloc[i])
                    buff_x_real.append(trough_x_real.iloc[i])
                    counter = 0
                else:
                    counter += 1
                if trough_y.iloc[i] < buff_y[-1]:
                    buff_x = [trough_x[i]]
                    buff_y = [trough_y.iloc[i]]
                    buff_x_real = [trough_x_real.iloc[i]]
                    counter = 0

                if counter >= counter_limit:
                    buff_x = [trough_x[i]]
                    buff_y = [trough_y.iloc[i]]
                    buff_x_real = [trough_x_real.iloc[i]]

            last_x = []
            last_y = []
            first_x = []
            first_y = []
            steps = []
            last_x_real = []
            for i in trend_x:
                last_y.append(0)
                last_x.append(0)
                first_y.append(0)
                first_x.append(0)
                steps.append(0)
                last_x_real.append(0)
            for i in range(0, len(trend_x)):
                first_y[i] = trend_y[i][0]
                first_x[i] = trend_x[i][0]
                last_x[i] = trend_x[i][-1]
                last_y[i] = trend_y[i][-1]
                last_x_real[i] = trend_x_real[i][-1]
                steps[i] = (first_y[i] - last_y[i]) / (last_x[i] - first_x[i])
            trend_line_x = []
            trend_line_y = []
            trend_line_x_real = []
            for i in range(0, len(trend_x)):
                p1 = np.array([first_x[i], first_y[i]])
                p2 = np.array([last_x[i], last_y[i]])
                slope = (p2[1] - p1[1]) / (p2[0] - p1[0])
                angle_radians = np.arctan(slope)
                angle_degrees = np.degrees(angle_radians)
                if degree_side_limit == "up":
                    if abs(angle_degrees) >= degree_limit:
                        trend_line_x.append([first_x[i]])
                        trend_line_y.append([first_y[i]])
                        trend_line_x_real.append([last_x_real[i]])
                        for j in range(first_x[i] + 1, last_x[i] + 1):
                            trend_line_x[-1].append(j)
                            trend_line_y[-1].append(trend_line_y[-1][-1] - steps[i])
                elif degree_side_limit == "dn":
                    if abs(angle_degrees) <= degree_limit:
                        trend_line_x.append([first_x[i]])
                        trend_line_y.append([first_y[i]])
                        trend_line_x_real.append([last_x_real[i]])
                        for j in range(first_x[i] + 1, last_x[i] + 1):
                            trend_line_x[-1].append(j)
                            trend_line_y[-1].append(trend_line_y[-1][-1] - steps[i])
            valid_trend_line_x = []
            valid_trend_line_y = []
            start = []
            band_arr = []
            for i in range(0, len(trend_line_y)):
                valid = True
                if trend_line_x[i][-1] - trend_line_x[i][0] < distance_limit:
                    valid = False
                else:
                    for j in range(1, len(trend_line_y[i]) - 2):
                        if closeprice.iloc[trend_line_x[i][j]] < trend_line_y[i][j]:
                            valid = False
                            break
                if valid:
                    step1 = abs(trend_line_y[i][0] - trend_line_y[i][1])
                    band_buff = band_perc
                    count = 0
                    start_index = trend_line_x_real[i][0]
                    for l in range(trend_line_x[i][-1] + 1, trend_line_x[i][-1] + 5000000):
                        if l + 1 < len(closeprice):
                            if closeprice.iloc[l + 1] >= trend_line_y[i][-1] + step1 - (
                                    (trend_line_y[i][-1] + step1) * (band_buff / 100)):
                                trend_line_y[i].append(trend_line_y[i][-1] + step1)
                                trend_line_x[i].append(l)
                                count += 1
                                band_buff += band_increase_step
                            else:
                                trend_line_y[i].append(trend_line_y[i][-1] + step1)
                                trend_line_x[i].append(l)
                                trend_line_y[i].append(trend_line_y[i][-1] + step1)
                                trend_line_x[i].append(l + 1)
                                count += 1
                                break
                        else:
                            trend_line_y[i].append(trend_line_y[i][-1] + step1)
                            trend_line_x[i].append(l)
                            count += 1
                            break
                    if count >= countinues_limit:
                        valid_trend_line_x.append(trend_line_x[i])
                        valid_trend_line_y.append(trend_line_y[i])
                        band_arr.append(band_buff)
                        start.append(start_index)
            upperband = []
            lowerband = []
            for i in range(0, len(valid_trend_line_y)):
                arrup = []
                arrdn = []
                band_buff = band_perc
                for j in range(0, len(valid_trend_line_y[i])):
                    arrup.append(valid_trend_line_y[i][j] + (valid_trend_line_y[i][j] * (band_buff / 100)))
                    arrdn.append(valid_trend_line_y[i][j] - (valid_trend_line_y[i][j] * (band_buff / 100)))
                    band_buff += band_increase_step
                upperband.append(arrup)
                lowerband.append(arrdn)
            return valid_trend_line_x, valid_trend_line_y, upperband, lowerband, start
        except Exception as er:
            print("Something Wrong in Bullish Trend Line : ",er)

    pivot_connect1 = model['pivot_connect1']
    counter_limit1 = model['counter_limit1']
    degree_limit1 = model['degree_limit1']
    distance_limit1 = model['distance_limit1']
    countinues_limit1 = model['countinues_limit1']
    swing_percentage1 = model['swing_percentage1']
    degree_side_limit1 = model['degree_side_limit1']
    band_perc1 = model['band_perc1']
    band_increase_step1 = model['band_increase_step1']

    peak_x, peak_y, trough_x, trough_y, peak_x_real, trough_x_real = Swings(ClosePrice=ClosePrice, HighPrice=HighPrice,
                                                                             LowPrice=LowPrice,
                                                                             perc=swing_percentage1)
    check_peak_x = False
    check_trough_x = False
    TrendLineBullish = []
    TrendLineBearish = []
    High_Pivots = []
    Low_Pivots = []
    if len(peak_x) >=1:
        High_Pivots = [peak_x, peak_y,peak_x_real]
    if len(trough_x) >=1:
        Low_Pivots = [trough_x, trough_y, trough_x_real]
    if len(peak_x) > pivot_connect1:
        try:
            xlbear, ylbear, upband_bear, lowband_bear, start_bear = trendline_bearish(
                peak_x=peak_x, peak_y=peak_y,
                peak_x_real=peak_x_real,
                closeprice=ClosePrice,
                pivot_connect=pivot_connect1,
                counter_limit=counter_limit1,
                degree_limit=degree_limit1,
                distance_limit=distance_limit1,
                countinues_limit=countinues_limit1,
                degree_side_limit=degree_side_limit1,
                band_perc=band_perc1,
                band_increase_step=band_increase_step1)
            if len(xlbear) >=1:
                TrendLineBearish = [xlbear, ylbear, upband_bear, lowband_bear, start_bear]
                print(f"Found {len(xlbear)-1} Bearish Trend Line")
                check_peak_x = True
            else:
                print("Found High Pivots But Cant Draw a Valid Bearish Trend Line, Change The Model Params")
        except Exception as error:
            print("Something Wrong in Bearish Trend Line - Error : ", error)
    else:
        print("Swing Percentage is too high, Cant find enough high pivots to draw Bearish trend line")

    if len(trough_x) > pivot_connect1:
        try:
            xlbull, ylbull, upband_bull, lowband_bull, start_bull = trendline_bullish(
                trough_x=trough_x, trough_y=trough_y,
                trough_x_real=trough_x_real,
                closeprice=ClosePrice,
                pivot_connect=pivot_connect1,
                counter_limit=counter_limit1,
                degree_limit=degree_limit1,
                distance_limit=distance_limit1,
                countinues_limit=countinues_limit1,
                degree_side_limit=degree_side_limit1,
                band_increase_step=band_increase_step1)
            if len(xlbull)>=1:
                TrendLineBullish = [xlbull, ylbull, upband_bull, lowband_bull, start_bull]
                print(f"Found {len(xlbull) - 1} Bearish Trend Line")
                check_trough_x = True
            else:
                print("Found Low Pivots But Cant Draw a Valid Bullish Trend Line, Change The Model Params")

        except Exception as error:
            print("Something Wrong in Bullish Trend Line - Error : ",error)
    else:
        print("Swing Percentage is too high, Cant find enough Low pivots to draw Bullish trend line")
    if plot:
        fig = make_subplots(rows=1, cols=1, shared_xaxes=True,
                            specs=[[{'type': 'candlestick'}]],
                            vertical_spacing=0.01,
                            row_heights=[1])
        fig.add_trace(go.Candlestick(name="Chart", x=Index,
                                     open=OpenPrice, high=HighPrice, low=LowPrice,
                                     close=ClosePrice), row=1, col=1)
        if len(peak_x)>1:
            fig.add_trace(go.Scatter(name="High Pivot : ", x=peak_x, y=peak_y, mode="markers",
                                     marker=dict(color="red", size=10)),
                          row=1, col=1)
        if len(trough_x) >1:
            fig.add_trace(go.Scatter(name="Low Pivots : ", x=trough_x, y=trough_y, mode="markers",
                                     marker=dict(color="green", size=10)),
                          row=1, col=1)
        if check_trough_x:
            for i in range(0, len(xlbull)):
                fig.add_trace(go.Scatter(name=f"Bullish TL:{i}", x=xlbull[i], y=ylbull[i], mode="lines",
                                         marker=dict(color="#32e6e0", size=2)),
                              row=1, col=1)
                fig.add_trace(go.Scatter(name=f"Upper Band:{i} ", x=xlbull[i], y=upband_bull[i], mode="lines",
                                         marker=dict(color="#32e6e0", size=2)),
                              row=1, col=1)
                fig.add_trace(
                    go.Scatter(name=f"Lower Band:{i}", x=xlbull[i], y=lowband_bull[i], mode="lines", fill='tonexty',
                               fillcolor='rgba(1,1,1,0.2)',
                               marker=dict(color="#32e6e0", size=2)),
                    row=1, col=1)
        if check_peak_x:
            for i in range(0, len(xlbear)):
                fig.add_trace(go.Scatter(name=f"Bearish TL:{i}", x=xlbear[i], y=ylbear[i], mode="lines",
                                         marker=dict(color="#f73eaa", size=2)),
                              row=1, col=1)
                fig.add_trace(go.Scatter(name=f"Upper Band:{i}", x=xlbear[i], y=upband_bear[i], mode="lines",
                                         marker=dict(color="#f73eaa", size=2)),
                              row=1, col=1)
                fig.add_trace(
                    go.Scatter(name=f"Lower Band:{i}", x=xlbear[i], y=lowband_bear[i], mode="lines", fill='tonexty',
                               fillcolor='rgba(1,1,1,0.2)',
                               marker=dict(color="#f73eaa", size=2)),
                    row=1, col=1)
        draft_template = go.layout.Template()
        fig.update_layout(
            template=draft_template,
            xaxis_rangeslider_visible=False,
            height=600,
            width=1200,
            margin=dict(
                l=0,
                r=0,
                b=0,
                t=0,
                pad=0),
            showlegend=True,
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            bargap=0,  # gap between bars of adjacent location coordinates
            bargroupgap=0
        )
        for i in range(2, 0, -1):  # starting with the last and stopping at 0
            fig.update_xaxes(row=i, col=1, rangeslider_visible=False)
        fig.update_xaxes(gridcolor="#DCDCDC", showspikes=True)
        fig.update_yaxes(gridcolor="#DCDCDC", automargin=True, showspikes=True, side="right")
        fig.show()
    return TrendLineBullish, TrendLineBearish, High_Pivots, Low_Pivots

#--------------------------------Example Usage
# _,_,_,_=draw_trendline(model,df['close'],df['high'],df['low'],df['open'],df.index,True)