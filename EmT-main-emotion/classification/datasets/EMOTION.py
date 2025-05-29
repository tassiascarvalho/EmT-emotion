import os
import numpy as np
import pandas as pd
from base.prepare_data import PrepareData
from sklearn.preprocessing import LabelEncoder

class EMOTION(PrepareData):
    def __init__(self, args):
        super(EMOTION, self).__init__(args)
        self.csv_path = args.data_path
        self.label_encoder = LabelEncoder()

    def create_dataset(self, subject_list=None, split=False, sub_split=False, feature=False, band_pass_first=False):
        df = pd.read_csv(self.csv_path)
        y = self.label_encoder.fit_transform(df['label'])
        X = df.drop(columns=['label']).values

        self.folder = os.path.join(
            f"./data_processed/data_{self.args.data_format}_{self.args.dataset}_{self.args.label_type}"
        )

        segment_length = self.args.segment * self.args.sampling_rate
        total_samples = X.shape[0]

        if total_samples < segment_length:
            raise ValueError(f"Segmento ({segment_length}) maior que total de amostras ({total_samples}).")

        num_trials = total_samples // segment_length
        if num_trials == 0:
            raise ValueError("Não há dados suficientes para nenhum trial.")

        data = X[:num_trials * segment_length].reshape(num_trials, segment_length, -1)
        data = np.expand_dims(data, axis=0)              # (1, trial, time, chan)
        data = np.expand_dims(data, axis=-1)             # (1, trial, time, chan, 1)

        labels = y[:num_trials * segment_length].reshape(num_trials, segment_length)
        labels = labels[:, 0].reshape(1, -1)

        if split:
            print("[AVISO] Segmentação ignorada: dados já estão prontos por linha.")

        if feature:
            data = self.get_features(data=data, feature_type=self.args.data_format)

        print('✅ EMOTION dataset preparado e salvo em:', self.folder)

        for i in subject_list:
            data_i = np.array(data).copy()
            label_i = np.array(labels)

            if label_i.ndim == 0:
                label_i = label_i.reshape(1)

            if i == 1:
                # Aplica ruído leve apenas a sub1
                noise = np.random.normal(0, 0.01, data_i.shape)
                data_i += noise

            self.save(data_i, label_i, i)
            print(f"✅ Salvou sub{i} com shape {data_i.shape} e labels {label_i.shape}")


