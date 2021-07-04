import requests

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.142 Safari/537.36'
}

i = 0
while i < 500:
    r = requests.post("https://tek.khabkrai.ru/events/Oprosy/1325", data={'option':'1330'}, headers=headers)
    i = i + 1
