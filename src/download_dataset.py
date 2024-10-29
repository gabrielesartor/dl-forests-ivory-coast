# import dependencies
import os.path
from os import path
import ee
import geemap
import wxee
import requests
from PIL import Image
import pandas as pd
import numpy as np
import xarray
import rioxarray
import matplotlib.pyplot as plt
from netCDF4 import num2date
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import io
import imageio
from IPython.display import Image, display
from ipywidgets import widgets, Layout, HBox

print(tf.__version__)
print(tf.test.gpu_device_name())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

#################### GEE Authentication ####################
ee.Authenticate()
# Initialize the library.
ee.Initialize(project='your-gee-project')
wxee.Initialize()
############################################################

#################### PATHS DEFINITION ####################
years = ee.List(['2018', '2019', '2020']) # overlap between sentinel and jaxa dataset

dataset_name = "dataset_02_mean_2019"
year = 1 # 2019

# dataset_name = "dataset_02_mean"
# year = 2 # 2020

test_name = "test_1"

at_work_dir = "your_root_dir"
datasets_dir = f"{at_work_dir}/datasets"
results_dir = f"{at_work_dir}/results"

current_dataset_dir = f"{datasets_dir}/{dataset_name}"
current_result_dir = f"{results_dir}/{dataset_name}"
current_test_result_dir = f"{results_dir}/{dataset_name}/{test_name}"

dataset_sent1_dir = f"{current_dataset_dir}/sentinel1"
download_sent1 = False

dataset_sent2_dir = f"{current_dataset_dir}/sentinel2"
download_sent2 = False

dataset_forest_dir = f"{current_dataset_dir}/forest"
download_forest = False

dataset_sent_cloud_dir = f"{current_dataset_dir}/sentinel2cloud"
download_sent_cloud = False

dataset_jrc_dir = f"{current_dataset_dir}/jrc"
download_jrc = False

dataset_noisy_map_dir = f"{current_dataset_dir}/noisy_map"
download_noisy_map = False

dataset_s2_cp_map_dir = f"{current_dataset_dir}/s2_cp"
download_s2_cp_map = True


#################### PATHS CREATION ####################
if path.exists(datasets_dir) == False: # DATASETS DIR
  os.mkdir(datasets_dir)
  print(f"Created dir: {datasets_dir}")

if path.exists(results_dir) == False: # RESULTS DIR
  os.mkdir(results_dir)
  print(f"Created dir: {results_dir}")

if path.exists(current_dataset_dir) == False: # CURRENT DATASET DIR
  os.mkdir(current_dataset_dir)
  print(f"Created dir: {current_dataset_dir}")

if path.exists(current_result_dir) == False:  # CURRENT RESULTS DIR
  os.mkdir(current_result_dir)
  print(f"Created dir: {current_result_dir}")

if path.exists(current_test_result_dir) == False:  # CURRENT TEST RESULTS DIR
  os.mkdir(current_test_result_dir)
  print(f"Created dir: {current_test_result_dir}")

if path.exists(dataset_sent1_dir) == False:  # SENTINEL1 DIR
  os.mkdir(dataset_sent1_dir)
  print(f"Created dir: {dataset_sent1_dir}")

if path.exists(dataset_sent2_dir) == False:  # SENTINEL2 DIR
  os.mkdir(dataset_sent2_dir)
  print(f"Created dir: {dataset_sent2_dir}")

if path.exists(dataset_forest_dir) == False:  # FNF DIR
  os.mkdir(dataset_forest_dir)
  print(f"Created dir: {dataset_forest_dir}")

if path.exists(dataset_jrc_dir) == False:  # JRC DIR
  os.mkdir(dataset_jrc_dir)
  print(f"Created dir: {dataset_jrc_dir}")

if path.exists(dataset_noisy_map_dir) == False:  # JRC DIR
  os.mkdir(dataset_noisy_map_dir)
  print(f"Created dir: {dataset_noisy_map_dir}")

if path.exists(dataset_s2_cp_map_dir) == False:  # S2_CP DIR
  os.mkdir(dataset_s2_cp_map_dir)
  print(f"Created dir: {dataset_s2_cp_map_dir}")

############################################################

#################### LOAD DATASET ####################
CI_borders = ee.FeatureCollection("USDOS/LSIB_SIMPLE/2017").filter(ee.Filter.eq('country_na','Cote d\'Ivoire'))
grid = ee.Geometry.BBox(-6.0969, 5.5697, -4.7731, 7.1474).coveringGrid('EPSG:4326', 2560)
print(f"Number of tiles: {grid.size()}")
roi = grid.geometry()
grid.size()

############################################################

#################### DEFINITION OF MAP CREATION FUNCTIONS ####################
def map_sentinel1(roi,year):
  sent1 = ee.ImageCollection("COPERNICUS/S1_GRD") #ATTENZIONE IN ALCUNE IMMAGINI NON C'è LA VV E DA ERRORE! DA SCRIVERE CON UN IF
  sent1 = sent1.filterDate(ee.String(years.get(year)).cat('-05-01'),ee.String(years.get(year)).cat('-09-30') ).filterBounds(CI_borders).select('VV','VH').mean() #.select(['HH','HV']) #.select('VV','VH','VV','HV') selected all
  sent1 = sent1.clip(roi)
  return sent1

def map_sentinel2(roi, year, perc_cloud = 20):
  # Sentinel-2
  # B4,3,2 RGB + B8 NIR 10m
  sent2 = ee.ImageCollection("COPERNICUS/S2_SR")
  sent2 = sent2.filterDate(ee.String(years.get(year)).cat('-05-01'),ee.String(years.get(year)).cat('-09-30') ).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', perc_cloud)).filterBounds(CI_borders).select(['B4','B3','B2','B8']).mean()
  sent2 = sent2.clip(roi)
  return sent2

def map_forest(roi, year):
  forest = ee.ImageCollection("JAXA/ALOS/PALSAR/YEARLY/FNF4")
  forest = forest.filterDate(ee.String(years.get(year)).cat('-01-01'),ee.String(years.get(year)).cat('-12-31') ).filterBounds(CI_borders).mosaic()
  forest = forest.clip(roi)
  return forest

def map_sent2_cloud(roi, year):
  s2_cloudless_col = (ee.ImageCollection('COPERNICUS/S2_CLOUD_PROBABILITY'))
  s2_cloudless_col = s2_cloudless_col.filterDate(ee.String(years.get(year)).cat('-05-01'),ee.String(years.get(year)).cat('-09-30') ).filterBounds(CI_borders).mean()
  s2_cloudless_col = s2_cloudless_col.clip(roi)
  return s2_cloudless_col

def map_jrc(roi, year):
  jrc = (ee.ImageCollection('JRC/GFC2020/V1'))
  jrc = jrc.filterDate(ee.String(years.get(year)).cat('-12-31') ).filterBounds(CI_borders).mean()
  jrc = jrc.clip(roi)
  return jrc

def noisy_map():
  noisy_map = ee.Image('projects/your-gee-project/assets/fnf_changes_2020')
  noisy_map = noisy_map.clip(roi)
  return noisy_map

def map_s2_cp(roi, year, perc_cloud = 20):
  # Sentinel-2
  # B4,3,2 RGB + B8 NIR 10m
  s2_cp = ee.ImageCollection("COPERNICUS/S2_SR")
  s2_cp = s2_cp.filterDate(ee.String(years.get(year)).cat('-05-01'),ee.String(years.get(year)).cat('-09-30') ).filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', perc_cloud)).filterBounds(CI_borders).select(['MSK_CLDPRB']).mean()
  s2_cp = s2_cp.clip(roi)
  return s2_cp

############################################################

#################### MAP GENERATION ####################
sent1 = map_sentinel1(roi, year)
sent2 = map_sentinel2(roi, year, 20)
forest = map_forest(roi, year)
cloud = map_sent2_cloud(roi, year)
jrc = map_jrc(roi, year)
noisy_map = noisy_map()
s2_cp = map_s2_cp(roi, year, 20)
############################################################

#################### MAPS DOWNLOAD ####################
def save_image_sent1_to_tiff(image, index):
  name_tif = f'{dataset_sent1_dir}/image_{index}.tiff'

  if path.exists(name_tif) == True:
    print(f"File {name_tif} already exists!")
    return None

  url = image.getDownloadUrl({
                'name': name_tif,
                'bands': ['VV','VH'],
                'region': ee.Feature(grid.toList(grid.size()).get(index)).geometry(),
                'scale':10,
                'nodata': 0,
                'masked': False,
                'format': 'GEO_TIFF',
                'filePerBand': False
            })
  response = requests.get(url)
  with open(name_tif, 'wb') as fd: #specificato da non salvare su gdrive
      print(f"Saving {name_tif}")
      fd.write(response.content)
      fd.close()

# save sentinel data
if download_sent1:
  for i in range(grid.size().getInfo()): # download all the tiles
    save_image_sent1_to_tiff(sent1, int(i))

def save_image_sent2_to_tiff(image, index):
  name_tif = f'{dataset_sent2_dir}/image_{index}.tiff'

  if path.exists(name_tif) == True:
    print(f"File {name_tif} already exists!")
    return None

  url = image.getDownloadUrl({
                'name': name_tif,
                'bands': ['B4','B3','B2','B8'],
                'region': ee.Feature(grid.toList(grid.size()).get(index)).geometry(),
                'scale': 10,
                'nodata': 0,
                'masked': False,
                'format': 'GEO_TIFF',
                'filePerBand': False
            })
  response = requests.get(url)
  with open(name_tif, 'wb') as fd: #specificato da non salvare su gdrive
      print(f"Saving {name_tif}")
      fd.write(response.content)
      fd.close()

# save sentine2 data
if download_sent2:
  for i in range(grid.size().getInfo()): # download all the tiles
    save_image_sent2_to_tiff(sent2, int(i))

# load forest/non-forest data
def save_label_to_tiff(image, index):
  name_tif = f'{dataset_forest_dir}/image_{index}.tiff'

  if path.exists(name_tif) == True:
    print(f"File {name_tif} already exists!")
    return None

  url = image.getDownloadUrl({
                'name': name_tif,
                'region': ee.Feature(grid.toList(grid.size()).get(index)).geometry(),
                'scale': 10,
                'nodata': 0,
                'masked': False,
                'format': 'GEO_TIFF',
                'filePerBand': False
            })
  response = requests.get(url)
  with open(name_tif, 'wb') as fd: #specificato da non salvare su gdrive
      print(f"Saving {name_tif}")
      fd.write(response.content)
      fd.close()

# save forest data
if download_forest:
  for i in range(grid.size().getInfo()): # download all the tiles
    save_label_to_tiff(forest, int(i))

def save_cloud_to_tiff(image, index):
  name_tif = f'{dataset_sent_cloud_dir}/image_{index}.tiff'

  if path.exists(name_tif) == True:  # CURRENT RESULTS DIR
    print(f"File {name_tif} already exists!")
    return None

  url = image.getDownloadUrl({
                'name': name_tif,
                'bands': ['probability'],
                'region': ee.Feature(grid.toList(grid.size()).get(index)).geometry(),
                'scale': 10,
                'nodata': 0,
                'masked': False,
                'format': 'GEO_TIFF',
                'filePerBand': False
            })
  response = requests.get(url)
  with open(name_tif, 'wb') as fd: #specificato da non salvare su gdrive
      print(f"Saving {name_tif}")
      fd.write(response.content)
      fd.close()

# save cloud data
if download_sent_cloud:
  for i in range(grid.size().getInfo()): # download all the tiles
    save_cloud_to_tiff(cloud, i)


def save_jrc_to_tiff(image, index):
  name_tif = f'{dataset_jrc_dir}/image_{index}.tiff'

  if path.exists(name_tif) == True:  # CURRENT RESULTS DIR
    print(f"File {name_tif} already exists!", flush=True)
    return None

  url = image.getDownloadUrl({
                'name': name_tif,
                'bands': ['Map'],
                'region': ee.Feature(grid.toList(grid.size()).get(index)).geometry(),
                'scale': 10,
                'nodata': 0,
                'masked': False,
                'format': 'GEO_TIFF',
                'filePerBand': False
            })
  response = requests.get(url)
  with open(name_tif, 'wb') as fd: #specificato da non salvare su gdrive
      print(f"Saving {name_tif}", flush=True)
      fd.write(response.content)
      fd.close()

# save forest data
if download_jrc:
  for i in range(grid.size().getInfo()): # download all the tiles
    save_jrc_to_tiff(jrc, i)


###### NOISY MAP
def save_noisy_map_to_tiff(image, index):
  name_tif = f'{dataset_noisy_map_dir}/image_{index}.tiff'

  if path.exists(name_tif) == True:  # CURRENT RESULTS DIR
    print(f"File {name_tif} already exists!", flush=True)
    return None

  url = image.getDownloadUrl({
                'name': name_tif,
                'bands': ['fnf'],
                'region': ee.Feature(grid.toList(grid.size()).get(index)).geometry(),
                'scale': 10,
                'nodata': 0,
                'masked': False,
                'format': 'GEO_TIFF',
                'filePerBand': False
            })
  response = requests.get(url)
  with open(name_tif, 'wb') as fd: #specificato da non salvare su gdrive
      print(f"Saving {name_tif}", flush=True)
      fd.write(response.content)
      fd.close()

# save forest data
if download_noisy_map:
  for i in range(grid.size().getInfo()): # download all the tiles
    save_noisy_map_to_tiff(noisy_map, i)
############################################################

###### CLOUD PROBABILITY MAP
def save_image_s2_cp_to_tiff(image, index):
  name_tif = f'{dataset_s2_cp_map_dir}/image_{index}.tiff'

  if path.exists(name_tif) == True:
    print(f"File {name_tif} already exists!")
    return None

  url = image.getDownloadUrl({
                'name': name_tif,
                'bands': ['MSK_CLDPRB'],
                'region': ee.Feature(grid.toList(grid.size()).get(index)).geometry(),
                'scale': 10,
                'nodata': 0,
                'masked': False,
                'format': 'GEO_TIFF',
                'filePerBand': False
            })
  response = requests.get(url)
  with open(name_tif, 'wb') as fd: #specificato da non salvare su gdrive
      print(f"Saving {name_tif}")
      fd.write(response.content)
      fd.close()

# save sentine2 data
if download_s2_cp_map:
  for i in range(grid.size().getInfo()): # download all the tiles
    save_image_s2_cp_to_tiff(s2_cp, int(i))
############################################################

print("Downloads: everything done.")