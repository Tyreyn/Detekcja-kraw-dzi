import cv2
CV_PI = 3.1415926535897932384626433832795

# ***** WCZYTANIE ZDJĘCIA *****
src = cv2.imread("Lymphocyte.jpg")
# ***** ROZMYCIE DLA LoG *****
# brak rozmycia (1,1)
# rozmycie powyzej >(1,1)
img = cv2.GaussianBlur(src,(1,1),0)

# ***** Filtr Laplace'a *****
img_rgbL = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img_rgbL, cv2.COLOR_RGB2GRAY)
l = cv2.Laplacian(img_gray,-1,3)
lap = cv2.cvtColor(l, cv2.COLOR_GRAY2RGB)
final_lap = cv2.add(lap , src)

# ***** Metoda Canny'ego *****
img_rgbC = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_grayC = cv2.cvtColor(img_rgbC, cv2.COLOR_RGB2GRAY)
# Progi dla procedury histerezy 
prog1 = 125
prog2 = 150
c = cv2.Canny(img_grayC,prog1,prog2,apertureSize = 3)
canny = cv2.cvtColor(c, cv2.COLOR_GRAY2RGB)
final_cany = cv2.add(img , canny)

# ***** Metoda Hougha detekcja za pomocą linii *****
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray,110,140,apertureSize = 3)
houg = edges
final_houg = src.copy()
# Aby zmienic parametry linii, wystarczy zmienić poszczegolne wartosci 
# pozwala na rysowanie linii tylko o okreslonej dlugosci minimalnej
prog = 10
# grubosc linii
rozmiarlinii = 1
# maksymalna możliwa odległość pomiędzy punktami będącymi na tej samej linii do połączenia ich 
odleglosc = 2 
# ************************************************************************************
linesP = cv2.HoughLinesP(c, 1, CV_PI / 180, prog, None, rozmiarlinii ,odleglosc)
if linesP is not None:
    for i in range(0, len(linesP)):
        l = linesP[i][0]
        cv2.line(final_houg, (l[0], l[1]), (l[2], l[3]), (0,0,255), 1, 8)

# ***** Metoda Hougha detekcja za pomocą linii *****
houg_circ = src.copy()
# minDist = odległość pomiędzy dwoma środkami koła
# param1 = próg jak w przypadku metody Canny'ego
# param2 = im większa liczba tym wybierane są największe możliwe koła 
circles = cv2.HoughCircles(img_gray, cv2.HOUGH_GRADIENT, 1, minDist = 30, param1=240, param2 = 5, minRadius=0, maxRadius=20)
if circles is not None:
    for i in circles[0, :]:
        # obwód
        cv2.circle(houg_circ, (i[0], i[1]), i[2], (255, 255, 255), 2)
        # środek
        cv2.circle(houg_circ, (i[0], i[1]), 1, (0, 0, 255), 1)

# ***** Wyświetlanie
cv2.imshow("org",img)
cv2.imshow("shape",c)
cv2.imshow("lap",houg_circ)
cv2.waitKey(0)
