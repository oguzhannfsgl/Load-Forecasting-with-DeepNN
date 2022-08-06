
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import argparse
from tqdm import tqdm
import holidays

from dataset import *
from models import *
from metric_n_loss import *
    
    
def main(args):
    IS_CUSTOM = True if args.model_name=="custom" else False
    
    if args.data_from=="TURKEY":
        df = "../data/demand_df_turkey.csv"
        holidays_ = holidays.TR()
    elif args.data_from=="SPAIN":
        df = "../data/demand_df_spain.csv"
        holidays_ = holidays.ESP()
        
        
    train_df, test_df, [load_min, load_max] = get_pd_dataset(df, args.data_from, holidays_, IS_CUSTOM)
    train_x, train_y, test_x, test_y = get_np_dataset(train_df, test_df, IS_CUSTOM)
    train_dataset = get_tf_dataset(train_x, train_y, args.input_n)
    test_dataset = get_tf_dataset(test_x, test_y, args.input_n)

    mape, rmse = get_unnormalized_mape(load_min, load_max), get_unnormalized_rmse(load_min, load_max)

    if args.model_name == "custom":
        model = CustomModel()
    elif args.model_name == "deepenergy":
        model = DeepEnergy()
    elif args.model_name == "seqmlp":
        model = SeqMlp()
    else:
        raise ValueError("model name should be one of these 3 strings: 'custom', 'seqmlp', 'deepenergy'")

    optim = tf.keras.optimizers.Adam()
    model.compile(loss="mae", metrics=["mape", mape, rmse], optimizer=optim)

    lr_reducer = tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=3, verbose=1)
    early_stopper = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, verbose=1, restore_best_weights=True)

    history = model.fit(train_dataset, validation_data=test_dataset, epochs=150, callbacks=[lr_reducer, early_stopper])

    
    
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_from", type=str, default="SPAIN", help="Name of the country which dataset comes from")
    parser.add_argument("--model_name", type=str, default='custom', help="One of 'custom', 'seqmlp', 'deepenergy' ")
    parser.add_argument("--input_n", type=int, default=24, help="Number of input time sequence. Default to 24h")
    parser.add_argument("--output_n", type=int, default=24, help="Number of output time sequence. Default to 24h")
    args = parser.parse_args()
        
    main(args)
    
    print("Parsed arguments are", args.data_from, args.is_custom, args.input_n, args.output_n)