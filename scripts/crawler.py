from google_images_download import google_images_download
response = google_images_download.googleimagesdownload()   #class instantiation

arguments = {"keywords":"Nicolas Cage","limit":5000,"print_urls":True,"chromedriver": '/home/toby/Documents/HollowFakes/data/chromedriver'}   #creating list of arguments
response.download(arguments)