#!/usr/bin/env python
# coding: utf-8

# # Подключаем нужные библиотеки

# In[2]:


import cv2
import face_recognition
 
# Получаем данные с устройства (веб камера у меня всего одна, поэтому в аргументах 0)
video_capture = cv2.VideoCapture(0)
 
# Инициализируем переменные
face_locations = []


# In[3]:


from sound import Sound
Sound.volume_up() # увеличим громкость на 2 единицы
current = Sound.current_volume() # текущая громкость, если кому-то нужно

volum_half=50  # 50% громкость
volum_full=100 # 100% громкость

Sound.volume_max() # выставляем сразу по максимуму


# Работа со временем

# In[4]:


# Подключаем модуль для работы со временем
import time
# Подключаем потоки
from threading import Thread
import threading


# Функция для работы с активностью мыши

# In[5]:


from pynput import mouse

def func_mouse():
        with mouse.Events() as events:
            for event in events:
                if event == mouse.Events.Scroll or mouse.Events.Click:
                    #print('Переместил мышку/нажал кнопку/скролл колесиком: {}\n'.format(event))
                    print('Делаю половину громкости: ', time.ctime())
                    Sound.volume_set(volum_half)
                    break


# In[6]:


# Делаем отдельную функцию с напоминанием
def not_find():
    #print("Cкрипт на 15 секунд начинается ", time.ctime())
    print('Делаю 100% громкости: ', time.ctime())
    #Sound.volume_set(volum_full)
    Sound.volume_max()
    
    # Секунды на выполнение
    #local_time = 15
    # Ждём нужное количество секунд, цикл в это время ничего не делает
    #time.sleep(local_time)
    
    # Вызываю функцию поиска действий по мышке
    func_mouse()
    #print("Cкрипт на 15 сек прошел")


# # А тут уже сам код

# In[ ]:


while True:
    ret, frame = video_capture.read()
    
    '''
    # Resize frame of video to 1/2 size for faster face recognition processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
    rgb_frame = small_frame[:, :, ::-1]
    '''

    rgb_frame = frame[:, :, ::-1]
    
    face_locations = face_recognition.face_locations(rgb_frame)
    
    number_of_face = len(face_locations)
    
    '''
    #print("Я нашел {} лицо(лица) в данном окне".format(number_of_face))
    #print("Я нашел {} лицо(лица) в данном окне".format(len(face_locations)))
    '''
    
    if number_of_face < 1:
        print("Я не нашел лицо/лица в данном окне, начинаю работу:", time.ctime())
        '''
        th = Thread(target=not_find, args=()) # Создаём новый поток
        th.start() # И запускаем его
        # Пока работает поток, выведем на экран через 10 секунд, что основной цикл в работе
        '''
        #time.sleep(5)
        print("Поток мыши заработал в основном цикле: ", time.ctime())
        
        #thread = threading.Timer(60, not_find)
        #thread.start()
        
        not_find()
        '''
        thread = threading.Timer(60, func_mouse)
        thread.start()
        print("Поток мыши заработал.\n")
        # Пока работает поток, выведем на экран через 10 секунд, что основной цикл в работе
        '''
        #time.sleep(10)
        print("Пока поток работает, основной цикл поиска лица в работе.\n")
    else:
        #все хорошо, за ПК кто-то есть
        print("Я нашел лицо/лица в данном окне в", time.ctime())
        Sound.volume_set(volum_half)
        
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()


# # Второй вариант

# In[7]:


while True:
    ret, frame = video_capture.read()
    
    rgb_frame = frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    
    number_of_face = len(face_locations)
    
    '''
    #print("Я нашел {} лицо(лица) в данном окне".format(number_of_face))
    #print("Я нашел {} лицо(лица) в данном окне".format(len(face_locations)))
    '''
    
    if number_of_face < 1:
        #print("Я не нашел лицо/лица в данном окне, начинаю работу")
        th = Thread(target=not_find, args=()) # Создаём новый поток
        th.start() # И запускаем его
        # Пока работает поток, выведем на экран через 10 секунд, что основной цикл в работе
        time.sleep(10)
        print("Пока поток работает, основной цикл поиска лица в работе.\n")
    else:
        #все хорошо, за ПК кто-то есть
        print("Я нашел лицо/лица в данном окне в", time.ctime())
    
    for top, right, bottom, left in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
    
    cv2.imshow('Video', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video_capture.release()
cv2.destroyAllWindows()

