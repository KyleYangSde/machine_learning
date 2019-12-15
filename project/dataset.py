import numpy as np
import pandas as pd
import os
from collections import defaultdict
import datetime
import pytz


class project_data():

    def __init__(self, root_path):
        self.root_path = root_path
        self.pre_student = ['u00', 'u01', 'u02', 'u03', 'u04', 'u05', 'u07', 'u08', 'u09', 'u10', 'u12', 'u13', 'u14',
                            'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u22', 'u23', 'u24', 'u27', 'u30', 'u31', 'u32',
                            'u33', 'u34', 'u35', 'u36', 'u39', 'u42', 'u43', 'u44', 'u45', 'u46', 'u47', 'u49', 'u50',
                            'u51', 'u52', 'u53', 'u56', 'u57', 'u58', 'u59']
        self.post_student = ['u00', 'u01', 'u02', 'u03', 'u04', 'u05', 'u07', 'u09', 'u10', 'u14', 'u15', 'u16', 'u17',
                             'u19', 'u20', 'u23', 'u24', 'u27', 'u30', 'u31', 'u32', 'u33', 'u34', 'u35', 'u36', 'u42',
                             'u43', 'u44', 'u45', 'u46', 'u47', 'u49', 'u51', 'u52', 'u53', 'u56', 'u59']
        self.tz = pytz.timezone('America/New_York')
        self.duration = 6_048_000

    def FlourishingScale(self):
        flour_path = os.path.join(self.root_path, 'Outputs/FlourishingScale.csv')
        flour_file = np.array(pd.read_csv(flour_path))
        flour = defaultdict(int)
        for r in flour_file:
            if r[1] == 'pre':
                continue
            elif r[1] == 'post':
                for c in range(2, 10):
                    if np.isnan(float(r[c])):
                        continue
                    flour[r[0]] += float(r[c])
        return flour

    def PANAS(self):
        panas_path = os.path.join(self.root_path, 'Outputs/panas.csv')
        panas_file = np.array(pd.read_csv(panas_path))
        pos_panas = defaultdict(int)
        neg_panas = defaultdict(int)
        positive_choices = [2, 5, 9, 10, 12, 13, 15, 16, 18]
        for r in panas_file:
            if r[1] == 'pre':
                continue
            elif r[1] == 'post':
                for c in range(2, 20):
                    if np.isnan(float(r[c])):
                        continue
                    if c in positive_choices:
                        pos_panas[r[0]] += float(r[c])
                    else:
                        neg_panas[r[0]] += float(r[c])
        return pos_panas, neg_panas

    def conversation_freq(self):
        conv_freq_day = defaultdict(float)
        conv_freq_eve = defaultdict(float)
        conv_freq_nig = defaultdict(float)
        csv_dir = os.path.join(self.root_path, 'Inputs/sensing/conversation')
        for s in self.pre_student:
            file_path = os.path.join(csv_dir, 'conversation_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path))
            duration = csv_file[-1][1] - csv_file[0][0]
            alpha = self.duration / duration
            # print(alpha)
            for r in csv_file:
                time = datetime.datetime.fromtimestamp(r[0], tz=self.tz)
                if 9 <= time.hour < 18:
                    conv_freq_day[s] += alpha
                elif time.hour >= 18:
                    conv_freq_eve[s] += alpha
                else:
                    conv_freq_nig[s] += alpha
        return conv_freq_day, conv_freq_eve, conv_freq_nig

    def conversation_dura(self):
        conv_dura_day = defaultdict(float)
        conv_dura_eve = defaultdict(float)
        conv_dura_nig = defaultdict(float)
        csv_dir = os.path.join(self.root_path, 'Inputs/sensing/conversation')
        for s in self.pre_student:
            file_path = os.path.join(csv_dir, 'conversation_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path))
            duration = csv_file[-1][1] - csv_file[0][0]
            alpha = self.duration / duration
            for r in csv_file:
                start_time = datetime.datetime.fromtimestamp(r[0], tz=self.tz)
                end_time = datetime.datetime.fromtimestamp(r[1], tz=self.tz)
                time = (end_time - start_time).seconds * alpha
                if 9 <= start_time.hour < 18:
                    conv_dura_day[s] += time
                elif start_time.hour >= 18:
                    conv_dura_eve[s] += time
                else:
                    conv_dura_nig[s] += time
        return conv_dura_day, conv_dura_eve, conv_dura_nig

    def co_location(self):
        csv_dir = os.path.join(self.root_path, 'Inputs/sensing/bluetooth')
        co_location = defaultdict(float)
        temp_list = []
        for s in self.pre_student:
            file_path = os.path.join(csv_dir, 'bt_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path))
            temp_dict = defaultdict(int)
            for r in csv_file:
                temp_dict[r[1]] += 1
            temp_list.append(temp_dict)
        for s, dic in zip(self.pre_student, temp_list):
            file_path = os.path.join(csv_dir, 'bt_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path))
            duration = csv_file[-1][0] - csv_file[0][0]
            alpha = self.duration / duration
            # print(s, alpha)
            for r in csv_file:
                if dic[r[1]] > 10:
                    co_location[s] += alpha
        return co_location

    def activity(self):
        activity_day = defaultdict(float)
        activity_eve = defaultdict(float)
        activity_nig = defaultdict(float)
        csv_dir = os.path.join(self.root_path, 'Inputs/sensing/activity')
        for s in self.pre_student:
            file_path = os.path.join(csv_dir, 'activity_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path))
            duration = csv_file[-1][0] - csv_file[0][0]
            alpha = self.duration / duration
            # print(s, alpha)
            for r in csv_file:
                if r[1] == 0:
                    continue
                time = datetime.datetime.fromtimestamp(r[0], tz=self.tz)
                if 9 <= time.hour < 18:
                    activity_day[s] += alpha
                elif time.hour >= 18:
                    activity_eve[s] += alpha
                else:
                    activity_nig[s] += alpha
        return activity_day, activity_eve, activity_nig

    def traveled_distance(self):
        distance_day = defaultdict(int)
        distance_eve = defaultdict(int)
        distance_nig = defaultdict(int)
        csv_dir = os.path.join(self.root_path, 'Inputs/sensing/gps')
        for s in self.pre_student:
            file_path = os.path.join(csv_dir, 'gps_' + s + '.csv')
            csv_file = np.array(pd.read_csv(file_path, index_col=False))
            x, y = 0, 0
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
                x, y = r[4], r[5]
        return distance_day, distance_eve, distance_nig

    def block_distance(self, x1, y1, x2, y2):
        return abs(x1 - x2) + abs(y1 - y2)

    def indoor_mobility(self):
        indoor_mobility_day = defaultdict(int)
        indoor_mobility_eve = defaultdict(int)
        indoor_mobility_nig = defaultdict(int)
        activity_dir = os.path.join(self.root_path, 'Inputs/sensing/activity')
        wifi_dir = os.path.join(self.root_path, 'Inputs/sensing/wifi')
        for s in self.pre_student:
            activity_path = os.path.join(activity_dir, 'activity_' + s + '.csv')
            activity_file = np.array(pd.read_csv(activity_path, index_col=False))
            wifi_path = os.path.join(wifi_dir, 'wifi_' + s + '.csv')
            wifi_file = np.array(pd.read_csv(wifi_path, index_col=False))
            start = min(activity_file[0][0], wifi_file[0][0])
            end = max(activity_file[-1][0], wifi_file[-1][0])
            temp = np.zeros((end - start + 3))
            for r in activity_file:
                if r[1] == 0:
                    temp[r[0] - start] = 0
                else:
                    temp[r[0] - start] = 1
                    temp[r[0] - start + 1] = 1
                    temp[r[0] - start + 2] = 1
            last_timestamp = 0
            for r in wifi_file:
                if r[0] == last_timestamp:
                    continue
                last_timestamp = r[0]
                if temp[r[0] - start] == 1:
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
    a,b = data.PANAS()
    print(a)
    print(b)

