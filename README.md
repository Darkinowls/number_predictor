# number_predictor

# Мета роботи:
Отримати початкові навички щодо створення штучних нейронних
мереж, що здатні виконувати прості логічні функції, та нейронних мереж, що
здатні прогнозувати часові ряди.

# Реузльтат:
У ході виконання лабораторної роботи було розроблена нейронна мережа для передбачення числового ряду.
Був вивчений метод навчання нейронної мережі backpropagation.
Освоєні безліч термінів машинного навчання для подальшого вивчення цієї сфери.
Створена нейронна мережа показала достатньо точний результат в першому тесті,
проте для числового ряду середня похибка складала 0.27, що може свідчити про нелінійність даного числового ряду.
Тобто, для кращого передбачення числового ряду потребується ускладнити алгоритми та, мабуть, виділити більше комп’ютерних ресурсів.


# Опис дії однієї (з конкретними значеннями) епохи при використанні алгоритму зворотного поширення (back propagation):

Вхідні дані :
X1 = 0.58
X2 = 3.38
X3 = 0.91
Вихідні дані : 
Y = 5.8
Початкове значення усіх ваг (зображені на малюнку нище) :
Wij = random(-0.1, 0.1)
Коефіцієнт навчання:
V = 0.01
Кількість нейронів в прихованому шарі:
NUM_NEURONS = 5
Функція активації: Сигмоїда з маштабом рівним 10


1.Етап переднього ходу
Загалом, формула для обчилення суми на вхід до нейрона матиме формулу:

Знаходимо суму на вхід для нейрона №4 (самий верхній нейрон у схованому шарі) за формулою :

S1 =  = x1 * w14 + x2 * w24 + x3 * w34 = 0.58 * 0.1 + 0 * 3.38 + 0.91 * (-0.05) = = 0.058 + -0.0455 = 0.0125

За допомогою суми знаходимо значення нейрону №4 :
 X4 = F(S4) =   5.03125

Знайдемо значення виходу інших нейронів для інші значення :

S2 = 0.58 * -0.05 + 0 * 3.38 + 0.91 * (-0.1) = -0.12
X5 = 4.70036

S3 = 0.58 * -0.05 + -0.05 * 3.38 + 0.91 * 0.1 = -0.107
X6 = 4.73275


S4 = 0.58 * -0.05 + 0.1 * 3.38 + 0.91 * (-0.05) = 0.2635
X7 = 5.65496

S5 = 0.58 * -0.05 + 0.1 * 3.38 + 0.91 * (-0.05) = 0.2635
X8 = 5.65496
 

Зі здобутих значень нейронів в прихованому шарі знайдемо значення на вхід на прогнозоване значення : 

S6 =  = 0.1 * 5.03125 + (-0.1) * 4.70036 + 0 * 4.73275 +0 * 5.65496 +(-0.05) * 5.65496 = -0.249659

X9 = 4.37907

2.Етап заднього ходу (backpropogation)
Знаходимо помилку прогнозу:
E = X9 – Y = 4.37907 – 5.8 = -1.420925

Знаходимо помилку нейронів за формулою :
D4 =  = - 1.420925 * 0.1 = - 0.1420925

D5 =  = - 1.420925 * (-0.1) = 0.1420925
D6 =  = - 1.420925 * 0 = 0
D7 =  = - 1.420925 * 0 = 0
D8 =  = - 1.420925 * -0.05 = 0.07104625

Знаходимо похідну сигмоїдальної функції:
 

Присвоюємо нові вагові коефіцієнти починаючи з вагів, що перед прихованим слоєм, за формулою:
wij = wij - V *   * Dj * xi 

w14 = w14 - V *   * D4 * x1 = 0.1 – 0.01 *  * (-0.1420925) * 0.58 = 0.100206
w24 = w24 - V *   * D4 * x2 = 0 – 0.01 *  * (-0.1420925) * 3.38 = 0.00120063
w34 = w34 - V *   * D4 * x3 = -0.05 – 0.01 *  * (-0.1420925) * 0.91 =          -0.0496768


w15 = w15 - V *   * D5 * x1 = -0.05 – 0.01 *  * 0.1420925 * 0.58 =              -0.0502053
w24 = w24 - V *   * D5 * x2 = 0 – 0.01 *  * 0.1420925 * 3.38 =                     -0.0011964
w34 = w34 - V *   * D5 * x3 = -0.1 – 0.01 *  * 0.1420925 * 0.91 =                 -0.100322


w16 = w16 - V *   * D6 * x1 = -0.05 – 0.01 *  * 0 * 0.58 = -0.05
w26 = w26 - V *   * D6 * x2 = -0.05 – 0.01 *  * 0 * 3.38 = -0.05
w36 = w36 - V *   * D6 * x3 = 0.1 – 0.01 *  * 0 * 0.91 = 0.1


w17 = -0.05 + 0 = -0.05
w27 = 0.1 + 0 = 0.1
w37 = -0.05 + 0 = -0.05

w18 = w16 - V *   * D6 * x1 = -0.05 – 0.01 *  * 0.07104625 * 0.58 =
-0.0501012
w28 = w26 - V *   * D6 * x2 = 0.1 – 0.01 *  * 0.07104625 * 3.38 = 
0.09941
w38 = w36 - V *   * D6 * x3 = -0.05 – 0.01 *  * 0.07104625 * 0.91 =
-0.0501589

Присвоємо значення вагам після прихованого шару :

W49 = w49 - V *   * E * x4 = 0.1 – 0.01 *  * (-1.420925) * 5.03125 = 0.117597
W59 = w59 - V *   * E * x5 = -0.1 – 0.01 *  * (-1.420925) * 4.70036 = 
-0.0835604
W69 = w69 - V *   * E * x6 = 0 – 0.01 *  * (-1.420925) * 4.73275 = 
0.0165529
W79 = w79 - V *   * E * x7 = 0 – 0.01 *  * (-1.420925) * 5.65496 = 
0.0197784
W89 = w79 - V *   * E * x7 = -0.05 – 0.01 *  * (-1.420925) * 5.65496 = 
-0.0302216

Епоха завершена
