import requests

lat = 42.443962
lon = -76.501884
api_key = "b47a7670105c694b23c5f55f10bc060d"

url = "https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&appid={api_key}".format(lat=lat, lon=lon, api_key=api_key)
ithaca = requests.get(url)

print(ithaca.json())