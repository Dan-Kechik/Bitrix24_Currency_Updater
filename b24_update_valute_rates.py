from fast_bitrix24 import Bitrix

import requests
import time
import os
import xml.etree.ElementTree as ET
import datetime
from datetime import date, timedelta
from dataclasses import dataclass
import dataclasses
import os
from matplotlib import pyplot as plt


CB_URL = "http://www.cbr.ru/scripts/XML_dynamic.asp"
DOLLAR_TAG = 'R01235'
SMS_ENDPOINT = r'https://api.exolve.ru/messaging/v1/SendSMS'


exolve_account_phone = os.environ['EXOLVE_PHONE']
manager_phone = os.environ['MANAGER_PHONE']
sms_api_key = os.environ['MTS_API_KEY']
bitrix_api_key = os.environ['BITRIX_API_KEY']


def send_SMS(send_str: str):
    payload = {'number': exolve_account_phone, 'destination': manager_phone, 'text': send_str}
    r = requests.post(SMS_ENDPOINT, headers={'Authorization': 'Bearer '+sms_api_key}, data=json.dumps(payload))
    print(r.text)
    return r.text, r.status_code


@dataclass
class Settings:
    currency: str = DOLLAR_TAG
    frame_length: int = 10  # days
    average_difference: float = 100  # percents
    weighted_difference: float = 100  # percents
    upper_threshold: float | None = 102
    lower_threshold: float | None = None
    jump_threshold: float = 5  # percents
    frame_validity_threshold = 0.6  # partial less 1


MESSAGE_DICTIONARY = {
    'jump': 'Резкое изменение курса валют.',
    'upper_exceeding': 'Выход за верхнюю границу коридора.',
    'lower_exceeding': 'Выход за нижнюю границу коридора.',
    'average_diff': 'Большая разность средних значений.'
}


@dataclass
class EventsList:
    jump: datetime.date | None = None
    lower_exceeding: datetime.date | None = None
    upper_exceeding: datetime.date | None = None
    average_diff: datetime.date | None = None

    def push_event(self, name: str, date: datetime.date, settings: Settings):
        assert name in [f.name for f in dataclasses.fields(self)]
        previous_date = self.__getattribute__(name)
        cutoff_date = date-timedelta(settings.frame_length)
        if previous_date is not None and previous_date < cutoff_date:
            self.__setattr__(name, None)
        previous_date = self.__getattribute__(name)
        if previous_date is None:
            self.__setattr__(name, date)
            #result = send_SMS(MESSAGE_DICTIONARY[name])
            #print(result)


def get_rates(date_req1: datetime.date | None = None, date_req2: datetime.date | None = None, currency: str = DOLLAR_TAG):
    """
    Запрашивает курс валют с сайта ЦБ РФ в фомате .xml;
    Парсит полученный .xml, добавляя нужную информацию ее в список;
    
    :return: text of response
    :rtype: str;
    """
    # http://www.cbr.ru/scripts/XML_dynamic.asp?date_req1=02/03/2001&date_req2=14/03/2001&VAL_NM_RQ=R01235
    if date_req1 is None:
        date_req1 = date.today()
    if date_req2 is None:
        date_req2 = date_req1
    
    params = {
        'date_req1': date_req1.strftime("%d/%m/%Y"),
        'date_req2': date_req2.strftime("%d/%m/%Y"),
        'VAL_NM_RQ': currency
    }

    # Запрашиваем и парсим данные
    response = requests.get(CB_URL, params)
    encoded_text = response.text
    return encoded_text


def open_xml_file(filename: str):
    with open(filename, "r", encoding="utf-8") as file:
        xml_parser = ET.XMLParser(encoding="utf-8")
        parsed = ET.parse(file, parser=xml_parser)
    return parsed.getroot()


def parser_from_string(text):
    xml_parser = ET.XMLParser(encoding="utf-8")
    parsed = ET.fromstring(text, parser=xml_parser)
    return parsed


def parse_to_dict(parsed):
    # Достаем нужную информацию о валютах и добавляем ее в список
    values, dates = [], []
    for rec in parsed.iter("Record"):
        nominal = rec.find("Nominal").text
        value = rec.find("Value").text.replace(",", ".")
        values.append(float(value)/float(nominal))
        date = rec.attrib['Date']
        dates.append(datetime.datetime.strptime(date, "%d.%m.%Y").date())
    return values, dates


def fetch_frame_values(date1, date2, currency=DOLLAR_TAG):
    xml_string = get_rates(date1, date2, currency)
    my_parser = parser_from_string(xml_string)
    ans = parse_to_dict(my_parser)
    return ans


def data_frame(days=10, base_date=None):
    if base_date is None:
        base_date = date.today()
    day_start = base_date - timedelta(days=days)
    return day_start, base_date


def is_frame_invalid(data: list[float], settings: Settings):
    return len(data) < settings.frame_length*settings.frame_validity_threshold


def unary_checks(data: list[float], settings: Settings, events: EventsList, current_date: datetime.date | None = None):
    if is_frame_invalid(data, settings):
        return
    if current_date is None:
        current_date = date.today()
    # Derivative
    for ai in range(1, len(data)):
        if (data[ai] - data[ai-1])/data[ai] > settings.jump_threshold/100:
            events.push_event('jump', current_date, settings)
    # Bounds
    if settings.upper_threshold:
        if any([d > settings.upper_threshold for d in data]):
            events.push_event('upper_exceeding', current_date, settings)
    if settings.lower_threshold:
        if any([d < settings.lower_threshold for d in data]):
            events.push_event('lower_exceeding', current_date, settings)


def binary_checks(data_curr: list[float], data_prev: list[float], settings: Settings, events: EventsList,
                 current_date: datetime.date | None = None):
    if is_frame_invalid(data_curr, settings) or is_frame_invalid(data_prev, settings):
        return
    if current_date is None:
        current_date = date.today()
    # Arithmetic average
    avg1 = sum(data_prev)/len(data_prev)
    avg2 = sum(data_curr)/len(data_curr)
    if abs(avg1 - avg2) > settings.average_difference*avg1/100:
        events.push_event('average_diff', current_date, settings)


BITRIX_CODE = 'm1ry6ydxz4beyhmp'


from fast_bitrix24 import Bitrix
BITRIX_URL = r'https://b24-d40p9s.bitrix24.ru/rest/1/'+BITRIX_CODE


def do_update(data):
    """
        Выполняет обновление данных курсов валют на портале Битрикс24;
        Возвращает значение предыдущего курса;

        :return: Previous dollar cource
        :rtype: float
    """
    
    # Запрашиваем список всех валют с портала
    endpoint = Bitrix(BITRIX_URL)
    currency_get = endpoint.get_all("crm.currency.list")

    # Отправляем запрос на обновление курса валют
    update_data = [
        {
            "ID": 'USD',
            "fields":
            {
                "AMOUNT": data[-1]
            }
        }
    ]
    endpoint.call("crm.currency.update", items=update_data)

    for cur in currency_get:
        if cur['CURRENCY'] == 'USD':
            return float(cur['AMOUNT']), endpoint
    raise


def send_hook_to_bitrix(event, data):
    past_cource, endpoint = do_update(data)
    ratio = data[-1]/past_cource
    prices = endpoint.get_all("catalog.price.list")

    # Обновляем цены
    for p in prices:
        p['price'] *= ratio

    # Формируем данные об обновляемых полях
    update_data = [
        {
            "id": item["id"],
            "fields":
                {
                    "price": item["price"],
                }
        }
        for item in prices
    ]
    endpoint.call("catalog.price.update", items=update_data)
    print('Finished updating')


import numpy as np
from matplotlib.widgets import Button
from matplotlib import gridspec


def plot_results(dates_earl, data_earl, dates_curr, data_curr, settings: Settings, events: EventsList):
    spec = gridspec.GridSpec(ncols=1, nrows=2, height_ratios=[2, 1])
    figure = plt.figure()
    ax = figure.add_subplot(spec[0])  # (2, 1, 1)
    ax.plot(dates_earl, data_earl, '-o', color='blue', label='Курс')
    ax.plot(dates_curr, data_curr, '-o', color='blue')
    ax.plot(dates_earl, np.ones_like(data_earl)*np.mean(data_earl), '--', color='red', label='Среднее значение')
    ax.plot(dates_curr, np.ones_like(data_curr)*np.mean(data_curr), '--', color='red')
    label = 'Границы коридора'
    if settings.lower_threshold:
        ax.plot(dates_curr, np.ones_like(data_curr)*settings.lower_threshold, '--', color='green', label=label)
        label = None
    if settings.upper_threshold:
        ax.plot(dates_curr, np.ones_like(data_curr)*settings.upper_threshold, '--', color='green', label=label)
    if events.jump is not None:
        ax.plot([events.jump, events.jump], ax.ylims())
    ax.legend()
    ax.tick_params("x", rotation=90)

    control_ax = figure.add_subplot(spec[1])  # (2, 1, 2)
    text = ''
    for field in dataclasses.fields(events):
        if events.__getattribute__(field.name) is not None:
            text += MESSAGE_DICTIONARY[field.name]+f'\n'
    if len(text) == 0:
        text = 'Никаких событий не зафиксировано.'
    control_ax.text(0, 0.7, text)
    apply_button = Button(control_ax, 'Принять изменения')
    apply_button.on_clicked(lambda x: send_hook_to_bitrix(x, data_curr))
    plt.tight_layout(pad=1)
    plt.savefig(r'D:\\3.png')
    plt.show()


def main():
    current_settings = Settings()
    events = EventsList()
    # Retrieve frame for unary operations
    dates = data_frame(days=current_settings.frame_length)
    data, dates_curr = fetch_frame_values(*dates)
    unary_checks(data, current_settings, events)
    # Retrieve frame for binary operations
    dates = data_frame(days=current_settings.frame_length, base_date=dates[0])
    data_prev, dates_earl = fetch_frame_values(*dates)
    binary_checks(data, data_prev, current_settings, events)
    plot_results(dates_earl, data_prev, dates_curr, data, current_settings, events)

if __name__ == "__main__":
    xml_string = get_rates(date(year=2025, month=1, day=1), date(year=2025, month=1, day=17), DOLLAR_TAG)
    my_parser = parser_from_string(xml_string)
    ans = parse_to_dict(my_parser)
    print(ans)
    main()
