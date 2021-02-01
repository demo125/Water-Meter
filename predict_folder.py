from data_loader.data_loaders import WaterMeterDataLoader, WaterMeterDataset
import os
from PIL import Image
import glob
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt
import ntpath
import pandas as pd
from absl.flags import FLAGS
from absl import app, flags
from torchvision.datasets import ImageFolder
from model.model import WaterMeterModel
import torch
import tqdm
from model.decode import feature_to_y

flags.DEFINE_string('input_folder', '/home/mde/python/Water-Meter/example-images/', 'path to folder containing jpgs for prediction')
flags.DEFINE_string('model_weights', '/home/mde/python/Water-Meter/saved/models/WMN_FCSRN/final_model-77acc.pth', 'path to model weightsfile')
flags.DEFINE_string('output_csv', './predictions/predictions.csv', 'csv with predictions')
flags.DEFINE_string('output_folder', './predictions', 'folder with visualized predictions, if None')
flags.DEFINE_string('default_image_folder', '../pytorch-rotation-decoupled-detector/images/fota_zip_test',
                    'folder with default images')
flags.DEFINE_integer('batch_size', 128, 'batch size used for prediction')
flags.DEFINE_integer('num_workers', 4, 'number of cores to use')
flags.DEFINE_integer('n_gpu', 2, 'number of gpus to use')
flags.DEFINE_integer('unknown_class', 11, 'class id for empty or not unrecognizable digit.(WaterMeterDataset.unknown_class)')



def load_model(checkpoint_path):
    print('loading model...')

    model = WaterMeterModel()
    checkpoint = torch.load(checkpoint_path)
    if FLAGS.n_gpu > 1:
        model = torch.nn.DataParallel(model)
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()
    return model, device


def replace_unknow_class(Y):
    Y = [ ''.join([str(y) if y != FLAGS.unknown_class else '-' for y in yarr]) for yarr in Y]
    return Y


def plot_predictions(img_and_predictions_and_img_name):
    img_path, img_output_path, predictions, color = img_and_predictions_and_img_name

    img = Image.open(img_path)
    width, height = img.size
    if width > 200:
        img = img.resize((int(width*0.3), int(height*0.3)))
    plt.imshow(img)
    title_obj = plt.title(predictions)
    plt.setp(title_obj, color=color)
    plt.savefig(img_output_path)


def predict(data_loader, model, device, output_csv):

    print('predicting...')

    predictions = dict()
    with torch.no_grad():
        for i, (imgs, img_names) in enumerate(tqdm.tqdm(data_loader)):
            imgs = imgs.to(device)

            model_outputs = model(imgs)
            predicted_classes, batch_probs = feature_to_y(model_outputs, return_probs=True)
            final_digits = replace_unknow_class(predicted_classes)

            default_image_names = []
            for name in img_names:
                extension = name.split('.')[-1]
                def_name = ''.join(name.split('__')[:-1]) + f'.{extension}'
                default_image_names.append(def_name)

            for img_name, default_img_name, digits, probs in zip(img_names, default_image_names, final_digits, batch_probs):
                if default_img_name not in predictions:
                    predictions[default_img_name] = []
                predictions[default_img_name].append((img_name, digits, probs))

    return predictions


def postprocess_predictions(prediction_dict, output_csv):

    def get_single_prediction_for_img(prediction_dict):

        prediction_dict_selected_index = dict()

        for default_img_name in prediction_dict:
            img_predictions = prediction_dict[default_img_name]
            known_digit_count = []
            for idx, (img_name, digits, probs) in enumerate(img_predictions):
                # print(idx, img_name, digits, np.round(probs, 2), np.median(probs))
                count = len(digits) - digits.count('-')
                known_digit_count.append(count)

            max_known_digit_count_idx = np.argmax(known_digit_count)
            max_known_digit_count = known_digit_count[max_known_digit_count_idx]

            if known_digit_count.count(max_known_digit_count) > 1:
                max_mean_probs = -np.inf
                max_mean_probs_idx = None
                for idx, (img_name, _, probs) in enumerate(img_predictions):
                    mean_probs = np.mean(probs)
                    if mean_probs > max_mean_probs:
                        max_mean_probs = mean_probs
                        max_mean_probs_idx = idx
                final_idx = max_mean_probs_idx
            else:
                final_idx = max_known_digit_count_idx
            prediction_dict_selected_index[default_img_name] = final_idx

        return prediction_dict_selected_index

    prediction_dict_selected_index = get_single_prediction_for_img(prediction_dict)

    df_final_predictions = pd.DataFrame(columns=['img_name', 'prediction'])

    if FLAGS.output_folder is not None and FLAGS.default_image_folder:
        os.makedirs(os.path.join(FLAGS.output_folder, 'imgs'), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.output_folder, 'bbs'), exist_ok=True)
        df_images_to_export = pd.DataFrame(columns=[
            'input_img_path',
            'output_img_path',
            'label',
            'color'
        ])
        df_export_idx = 0

    for i, img_name in enumerate(tqdm.tqdm(prediction_dict)):

        selected_img_idx = prediction_dict_selected_index[img_name]

        final_img_prediction = prediction_dict[img_name][selected_img_idx]
        _, final_digits, _ = final_img_prediction

        df_final_predictions.loc[i] =[img_name, final_digits]

        
        if  FLAGS.output_folder is not None:
            #final prediction visualized on default image
            df_images_to_export.loc[df_export_idx] = [
                os.path.join(FLAGS.default_image_folder, img_name),
                os.path.join(FLAGS.output_folder,'imgs', img_name),
                final_digits,
                'green'
            ]
            df_export_idx += 1
            
            #predictions visualized on every cropped subimg of default image
            for idx, (croped_img_name, cropped_img_prediction, _) in enumerate(prediction_dict[img_name]):
                
                df_images_to_export.loc[df_export_idx] = [
                    os.path.join(FLAGS.input_folder, croped_img_name),
                    os.path.join(FLAGS.output_folder,'bbs', croped_img_name),
                    cropped_img_prediction,
                    'green' if idx == selected_img_idx else 'red'
                ]
                df_export_idx += 1
                

    if FLAGS.output_folder is not None and FLAGS.default_image_folder:
        print(f"exporting imgs with predictions to {FLAGS.output_folder}")
        with Pool(processes=FLAGS.num_workers) as pool:
            pool.map(plot_predictions, df_images_to_export.values.tolist())

    print(f'predictions written to {output_csv}')

    df_final_predictions.to_csv(output_csv, index=False, sep=';')

import pickle
def main(_args):

        data_loader = WaterMeterDataLoader(
            data_dir=FLAGS.input_folder,
            batch_size=FLAGS.batch_size,
            shuffle=False,
            num_workers=FLAGS.num_workers,
            validation_split=0,
            mode='predict'
        )
        
        model, device = load_model(FLAGS.model_weights)
        
        prediction_dict = predict(data_loader, model, device, FLAGS.output_csv)

        postprocess_predictions(prediction_dict, FLAGS.output_csv)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
