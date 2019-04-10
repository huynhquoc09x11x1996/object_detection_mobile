import requests

api_key = 'acc_275de35a8579001'
api_secret = '49ae9ef9f39a6433f79bb0135b61393e'
image_path = '/path/to/your/image.jpg'

response = requests.post('https://api.imagga.com/v2/tags?limit=5',
                         auth=(api_key, api_secret),
                         files={'image': open("img_detected/object-detection-8.jpg", 'rb')})
print(response.json())
