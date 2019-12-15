import numpy as np
import pandas as pd
import os
from collections import defaultdict
import datetime
import pytz
import matplotlib.pyplot as plt


class project_data():
    '''
    the class generate the features might be used
    the root path of input and output is needed
    all function in the class return dictionary(s)
    '''
    def __init__(self, root_path):
        self.root_path = root_path
        self.pre_student = ['u00', 'u01', 'u02', 'u03', 'u04', 'u05', 'u07', 'u08', 'u09', 'u10', 'u12', 'u13', 'u14',
                            'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u22', 'u23', 'u24', 'u27', 'u30', 'u31', 'u32',
                            'u33', 'u34', 'u35', 'u36', 'u39', 'u42', 'u43', 'u44', 'u45', 'u46', 'u47', 'u49', 'u50',
                            'u51', 'u52', 'u53', 'u56', 'u57', 'u58', 'u59']
        self.post_student = ['u00', 'u01', 'u02', 'u03', 'u04', 'u05', 'u07', 'u09', 'u10', 'u14', 'u15', 'u16', 'u17',
                             'u19', 'u20', 'u23', 'u24', 'u27', 'u30', 'u31', 'u32', 'u33', 'u34', 'u35', 'u36', 'u42',
                             'u43', 'u44', 'u45', 'u46', 'u47', 'u49', 'u51', 'u52', 'u53', 'u56', 'u59']
        self.tz = pytz.timezone('America/New_York')  # the time zone of Dartmouth College
        self.duration = 6_048_000  # the total seconds of 10 weeks

    def FlourishingScale(self):
        '''
        the flourishing scale, as well as the panas scale are miss some values.
        for the fairness and calculation, we add a median value if there is a nan since the missing value may suggest
        that it's hard to say it negative or positive. but it's unappropriate to add zero.
        :returns: dictionary key: student id, value: sum of all choice
        '''
        flour_path = os.path.join(self.root_path, 'Outputs/FlourishingScale.csv')
        flour_file = np.array(pd.read_csv(flour_path))
        pre_flour = defaultdict(int)
        post_flour = defaultdict(int)
        for r in flour_file:
            if r[1] == 'pre':
                for c in range(2, 10):
                    if np.isnan(float(r[c])):  # one of several ways to detect nan
                        pre_flour[r[0]] += 4  # the choice is 1-7 so the median value we use is 4
                    else:
                        pre_flour[r[0]] += float(r[c])
            elif r[1] == 'post':
                for c in range(2, 10):
                    if np.isnan(float(r[c])):
                        pre_flour[r[0]] += 4
                    else:
                        post_flour[r[0]] += float(r[c])
        return pre_flour, post_flour

    def PANAS(self):
        '''
        similar above
        :return: 4 dictionaries
        '''
        panas_path = os.path.join(self.root_path, 'Outputs/panas.csv')
        panas_file = np.array(pd.read_csv(panas_path))
        pre_pos_panas = defaultdict(int)
        pre_neg_panas = defaultdict(int)
        post_pos_panas = defaultdict(int)
        post_neg_panas = defaultdict(int)
        positive_choices = [2, 5, 9, 10, 12, 13, 15, 16, 18]  # the positive and negatives value are mixed and are different from the original one
        for r in panas_file:
            if r[1] == 'pre':
                for c in range(2, 20):
                    if c in positive_choices:
                        if np.isnan(float(r[c])):
                            pre_pos_panas[r[0]] += 3  # the choice of panas is 1-5 so the median value is 3
                        else:
                            pre_pos_panas[r[0]] += float(r[c])
                    else:
                        if np.isnan(float(r[c])):
                            pre_neg_panas[r[0]] += 3
                        else:
                            pre_neg_panas[r[0]] += float(r[c])
            elif r[1] == 'post':
                for c in range(2, 20):
                    if c in positive_choices:
                        if np.isnan(float(r[c])):
                            post_pos_panas[r[0]] += 3
                        else:
                            post_pos_panas[r[0]] += float(r[c])
                    else:
                        if np.isnan(float(r[c])):
                            post_neg_panas[r[0]] += 3
                        else:
                            post_neg_panas[r[0]] += float(r[c])
        return pre_pos_panas, pre_neg_panas, post_pos_panas, post_neg_panas

    def conversation_freq(self):
        '''
        as the reference required, the data is divided into three time period;
        day: 9am - 6pm
        evening: 6pm - 12pm
        night: 12pm - 9am
        in the data, some sensing data are appearntly short than the other, just keep it
        '''
        conv_freq_day = defaultdict(float)
        conv_freq_eve = defaultdict(float)
        conv_freq_nig = defaultdict(float)
        csv_dir = os.path.join(self.root_path, 'Inputs/sensing/conversation')
        for s in self.pre_student:
            file_path = os.path.join(csv_dir, 'conversation_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path))
            for r in csv_file:
                time = datetime.datetime.fromtimestamp(r[0], tz=self.tz) # change to the time zone tz=
                if 9 <= time.hour < 18:
                    conv_freq_day[s] += 1
                elif time.hour >= 18:
                    conv_freq_eve[s] += 1
                else:
                    conv_freq_nig[s] += 1
        return conv_freq_day, conv_freq_eve, conv_freq_nig

    def conversation_dura(self):
        '''
        the each conversation belong to the time period of the start timestamp
        day: 9am - 6pm
        evening: 6pm - 12pm
        night: 12pm - 9am
        '''
        conv_dura_day = defaultdict(float)
        conv_dura_eve = defaultdict(float)
        conv_dura_nig = defaultdict(float)
        csv_dir = os.path.join(self.root_path, 'Inputs/sensing/conversation')
        for s in self.pre_student:
            file_path = os.path.join(csv_dir, 'conversation_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path))
            for r in csv_file:
                start_time = datetime.datetime.fromtimestamp(r[0], tz=self.tz)  # change to the time zone tz=
                end_time = datetime.datetime.fromtimestamp(r[1], tz=self.tz)
                time = (end_time - start_time).seconds
                if 9 <= start_time.hour < 18:
                    conv_dura_day[s] += time
                elif start_time.hour >= 18:
                    conv_dura_eve[s] += time
                else:
                    conv_dura_nig[s] += time
        return conv_dura_day, conv_dura_eve, conv_dura_nig

    def co_location(self):
        '''
        the bluetooth is used to calculate co-location
        some data appeared less than 10 times should be removed
        '''
        csv_dir = os.path.join(self.root_path, 'Inputs/sensing/bluetooth')
        co_location = defaultdict(float)
        temp_list = []  # used to save the dictionaries
        for s in self.pre_student:
            file_path = os.path.join(csv_dir, 'bt_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path))
            temp_dict = defaultdict(int)  # used to save how many times the bluetooth mac has meet
            for r in csv_file:
                temp_dict[r[1]] += 1
            temp_list.append(temp_dict)
        for s, dic in zip(self.pre_student, temp_list):
            file_path = os.path.join(csv_dir, 'bt_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path))
            for r in csv_file:
                if dic[r[1]] > 10:
                    co_location[s] += 1
        return co_location

    def activity(self):
        '''
        the csv files are of different length, so the result will divide by the length of 0 value
        '''
        activity_day = defaultdict(float)
        activity_eve = defaultdict(float)
        activity_nig = defaultdict(float)
        csv_dir = os.path.join(self.root_path, 'Inputs/sensing/activity')
        for s in self.pre_student:
            file_path = os.path.join(csv_dir, 'activity_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path))
            zeros = 0
            for r in csv_file:
                if r[1] == 0:
                    zeros += 1
                    continue
                time = datetime.datetime.fromtimestamp(r[0], tz=self.tz)
                if 9 <= time.hour < 18:
                    activity_day[s] += 1
                elif time.hour >= 18:
                    activity_eve[s] += 1
                else:
                    activity_nig[s] += 1
            activity_day[s] /= zeros
            activity_eve[s] /= zeros
            activity_nig[s] /= zeros
        return activity_day, activity_eve, activity_nig

    def traveled_distance(self):
        '''
        use gps value to calculate traveled distance
        the change of  latitude and longitude represent the distance
        block distance is used
        '''
        distance_day = defaultdict(int)
        distance_eve = defaultdict(int)
        distance_nig = defaultdict(int)
        csv_dir = os.path.join(self.root_path, 'Inputs/sensing/gps')
        for s in self.pre_student:
            file_path = os.path.join(csv_dir, 'gps_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path, index_col=False))
            x, y = 0, 0  # last
            for r in csv_file:
                if x == 0 and y == 0:
                    x, y = r[4], r[5]
                    continue
                time = datetime.datetime.fromtimestamp(r[0], tz=self.tz)
                if 9 <= time.hour < 18:
                    distance_day[s] += self.block_distance(x, y, r[4], r[5])
                elif time.hour >= 18:
                    distance_eve[s] += self.block_distance(x, y, r[4], r[5])
                else:
                    distance_nig[s] += self.block_distance(x, y, r[4], r[5])
                x, y = r[4], r[5]  # record last gps
        return distance_day, distance_eve, distance_nig

    def block_distance(self, x1, y1, x2, y2):
        # block distace
        return abs(x1 - x2) + abs(y1 - y2)

    def indoor_mobility(self):
        '''
        wifi scan logs indicate indoor mobility
        '''
        indoor_mobility_day = defaultdict(int)
        indoor_mobility_eve = defaultdict(int)
        indoor_mobility_nig = defaultdict(int)
        wifi_dir = os.path.join(self.root_path, 'Inputs/sensing/wifi')
        for s in self.pre_student:
            wifi_path = os.path.join(wifi_dir, 'wifi_' + s + '.csv')
            wifi_file = np.array(pd.read_csv(wifi_path, index_col=False))
            last_timestamp = 0  # wifi may detect several wifi at same time
            for r in wifi_file:
                if r[0] == last_timestamp:
                    continue
                last_timestamp = r[0]
                time = datetime.datetime.fromtimestamp(r[0], tz=self.tz)
                if 9 <= time.hour < 18:
                    indoor_mobility_day[s] += 1
                elif time.hour >= 18:
                    indoor_mobility_eve[s] += 1
                else:
                    indoor_mobility_nig[s] += 1
        return indoor_mobility_day, indoor_mobility_eve, indoor_mobility_nig

    def sleep_duration(self):
        sleep = defaultdict(list)
        activity_dir = os.path.join(self.root_path, 'Inputs/sensing/activity')
        audio_dir = os.path.join(self.root_path, 'Inputs/sensing/audio')
        phonecharge_dir = os.path.join(self.root_path, 'Inputs/sensing/phonecharge')
        phonelock_dir = os.path.join(self.root_path, 'Inputs/sensing/phonelock')
        for s in self.pre_student:
            activity_path = os.path.join(activity_dir, 'activity_' + s + '.csv')
            activity_file = np.array(pd.read_csv(activity_path, index_col=False))
            audio_path = os.path.join(audio_dir, 'audio_' + s + '.csv')
            audio_file = np.array(pd.read_csv(audio_path, index_col=False))
            phonecharge_path = os.path.join(phonecharge_dir, 'phonecharge_' + s + '.csv')
            phonecharge_file = np.array(pd.read_csv(phonecharge_path, index_col=False))
            phonelock_path = os.path.join(phonelock_dir, 'phonelock_' + s + '.csv')
            phonelock_file = np.array(pd.read_csv(phonelock_path, index_col=False))
            start = min(activity_file[0][0], audio_file[0][0], phonelock_file[0][0], phonecharge_file[0][0])
            end = max(activity_file[-1][0], audio_file[-1][0], phonelock_file[-1][0], phonecharge_file[-1][0])
            temp = np.ones((end - start + 3, 4), dtype=np.float32)
            temp[:, 0] = temp[:, 0] * 0.5445
            temp[:, 1] = temp[:, 1] * 0.3484
            temp[:, 2] = temp[:, 2] * 0
            temp[:, 3] = temp[:, 3] * 0
            for r in activity_file:
                if r[1] != 0:
                    temp[r[0] - start][0] = 0
                    temp[r[0] - start + 1][0] = 0
                    temp[r[0] - start + 2][0] = 0
                else:
                    temp[r[0] - start][0] = 0.5445
            for r in audio_file:
                if r[1] != 0:
                    temp[r[0] - start][1] = 0
                    temp[r[0] - start + 1][1] = 0
                    temp[r[0] - start + 2][1] = 0
                else:
                    temp[r[0] - start][1] = 0.3484
            for r in phonelock_file:
                for rr in range(r[0], r[1]):
                    temp[rr - start][2] = 0.0512
            for r in phonecharge_file:
                for rr in range(r[0], r[1]):
                    temp[rr - start][3] = 0.0469
            time = 0
            for r in temp:
                if np.sum(r) > 0.9:
                    time += 1
            sleep[s] = time / (end - start)
            print(s)
        return sleep


if __name__ == '__main__':
    print()
    root_path = os.path.join(os.path.dirname(__file__), 'StudentLife_Dataset')
    data = project_data(root_path=root_path)
    #data.sleep_duration()
    '''
    indoor_mobility_day, indoor_mobility_eve, indoor_mobility_nig = data.indoor_mobility()
    print(indoor_mobility_day)
    print(indoor_mobility_eve)
    print(indoor_mobility_nig)
    '''
    '''
    activity_day, activity_eve, activity_nig = data.activity()
    print(activity_day)
    print(activity_eve)
    print(activity_nig)
    '''
    
    co_location = data.co_location()
    print(co_location)
    
    '''
    conv_freq_day, conv_freq_eve, conv_freq_nig = data.conversation_freq()
    print(conv_freq_day)
    print(conv_freq_eve)
    print(conv_freq_nig)
    '''
    '''
    conv_dura_day, conv_dura_eve, conv_dura_nig = data.conversation_dura()
    print(conv_dura_day)
    print(conv_dura_eve)
    print(conv_dura_nig)
    '''


