# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'test.ui'
#
# Created by: PyQt5 UI code generator 5.15.1
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.

from PyQt5 import *

from PyQt5 import QtCore, QtGui, QtWidgets, Qt
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog

from PyQt5.QtCore import * 
from PyQt5.QtGui import * 
from PyQt5.QtWidgets import * 


import sys







##############################################################


class ExpandWidget(QtWidgets.QWidget) :

    def __init__(self, *args, **kwargs):
        super(ExpandWidget, self).__init__(*args, **kwargs)
        
        self.setStyleSheet("border: 1px solid #000000; background: #C4C4C1")

        self.setMinimumSize(330, 100)
        self.Header = ExpandWidgetHeader(self)
        self.Header.mousePressEvent = self.hideBody         

        self.Body = QtWidgets.QWidget(self)
        self.Body.setGeometry(0, 25, 330, 400)
        self.Body.hide()
        self.setMinimumSize(330, 25)

        #QtCore.QObject.connect(self.Header, SIGNAL(headerClicked(bool)),self,SLOT(hideBody(bool)))

    def hideBody(self,eve):
        if not self.Header.m_isHidden : 
            self.Body.hide()
            self.setMinimumHeight(self.Header.height())
            
        else :
            self.setMinimumHeight(self.Body.height() + self.Header.height())
            self.Body.show()
            
        self.Header.m_isHidden = not self.Header.m_isHidden
        
class ExpandWidgetHeader (QtWidgets.QLabel):
    
    def __init__(self, parent):
        super(ExpandWidgetHeader,self).__init__(parent)
        self.m_isHidden= True
        self.setText("Text")
        self.setStyleSheet("border: 1px solid #000000; background: #898983")
        self.setGeometry(0, 0, 330, 25)
        self.setAlignment(Qt.AlignCenter)

    
##############################################################

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):

        # -- Main Window --
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1500, 822)
        MainWindow.setAutoFillBackground(True)

        # -- central widget --
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")

        # -- ImageLayout --
        self.ImageLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.ImageLayoutWidget.setGeometry(QtCore.QRect(-1, -1, 801, 551))
        self.ImageLayoutWidget.setObjectName("ImageLayoutWidget")
        self.ImageLayoutWidget.setStyleSheet("border: 1px solid #000000; background: #C4C4C1")

        self.ImageLayout = QtWidgets.QHBoxLayout(self.ImageLayoutWidget)
        self.ImageLayout.setContentsMargins(0, 0, 0, 0)
        self.ImageLayout.setObjectName("ImageLayout")

        self.setupLabelImageUi(MainWindow)
        # ---
        self.setupToolBarUi(MainWindow)

        # Button Layout
        self.ButtonLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.ButtonLayoutWidget.setGeometry(QtCore.QRect(-1, 569, 1001, 201))
        self.ButtonLayoutWidget.setObjectName("ButtonLayoutWidget")
        self.ButtonLayout = QtWidgets.QHBoxLayout(self.ButtonLayoutWidget)
        self.ButtonLayout.setContentsMargins(30, 0, 30, 0)
        self.ButtonLayout.setSpacing(30)
        self.ButtonLayout.setObjectName("ButtonLayout")
        self.setupButtonUi(MainWindow)



        MainWindow.setCentralWidget(self.centralwidget)

        self.setupMenuBar(MainWindow)

        self.retranslateUi(MainWindow)

        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        



    # ---- Setup Ui Fonction ----
    def setupLabelImageUi(self, MainWindow):
        # --  --
        self.label_image_original = QtWidgets.QLabel(self.ImageLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setKerning(True)
        self.label_image_original.setFont(font)
        self.label_image_original.setAlignment(QtCore.Qt.AlignCenter)
        self.label_image_original.setObjectName("label_image_original")
        self.ImageLayout.addWidget(self.label_image_original)


        self.label_image_pixel = QtWidgets.QLabel(self.ImageLayoutWidget)
        font = QtGui.QFont()
        font.setBold(True)
        font.setItalic(False)
        font.setWeight(75)
        font.setKerning(True)
        self.label_image_pixel.setFont(font)
        self.label_image_pixel.setAlignment(QtCore.Qt.AlignCenter)
        self.label_image_pixel.setObjectName("label_image_pixel")
        self.ImageLayout.addWidget(self.label_image_pixel)
        
        print(self.label_image_pixel.size())
        #print(self.label_image_original.size())


    def setupButtonUi(self, MainWindow):
        # --  --
        self.pushButton_ouvrir = QtWidgets.QPushButton(self.ButtonLayoutWidget)
        self.pushButton_ouvrir.setMinimumSize(QtCore.QSize(0, 70))
        self.pushButton_ouvrir.setObjectName("pushButton_ouvrir")
        
        self.pushButton_ouvrir.clicked.connect(self.openImage)
        self.ButtonLayout.addWidget(self.pushButton_ouvrir)

        # --  --
        self.pushButton_generer = QtWidgets.QPushButton(self.ButtonLayoutWidget)
        self.pushButton_generer.setMinimumSize(QtCore.QSize(0, 70))
        self.pushButton_generer.setObjectName("pushButton_generer")
		
        self.pushButton_generer.clicked.connect(self.generateImage)
        
        self.ButtonLayout.addWidget(self.pushButton_generer)

        # --  --
        self.pushButton_exporter = QtWidgets.QPushButton(self.ButtonLayoutWidget)
        self.pushButton_exporter.setMinimumSize(QtCore.QSize(0, 70))
        self.pushButton_exporter.setObjectName("pushButton_exporter")
        self.pushButton_exporter.clicked.connect(self.exportImage)

        self.ButtonLayout.addWidget(self.pushButton_exporter)


    def setupMenuBar(self, MainWindow):
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 1014, 21))
        self.menubar.setObjectName("menubar")
        self.menuFichier = QtWidgets.QMenu(self.menubar)
        self.menuFichier.setObjectName("menuFichier")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.actionNouveau = QtWidgets.QAction(MainWindow)
        self.actionNouveau.setObjectName("actionNouveau")
        self.actionOuvrir = QtWidgets.QAction(MainWindow)
        self.actionOuvrir.setObjectName("actionOuvrir")
        self.actionOuvrir_une_image = QtWidgets.QAction(MainWindow)
        self.actionOuvrir_une_image.setObjectName("actionOuvrir_une_image")
        self.actionEnregistrer = QtWidgets.QAction(MainWindow)
        self.actionEnregistrer.setObjectName("actionEnregistrer")
        self.actionEnregistrer_une_copie = QtWidgets.QAction(MainWindow)
        self.actionEnregistrer_une_copie.setObjectName("actionEnregistrer_une_copie")
        self.actionImporter_une_Palette = QtWidgets.QAction(MainWindow)
        self.actionImporter_une_Palette.setObjectName("actionImporter_une_Palette")
        self.actionExporter_une_Palette = QtWidgets.QAction(MainWindow)
        self.actionExporter_une_Palette.setObjectName("actionExporter_une_Palette")
        self.actionExporter_Pixel_Art = QtWidgets.QAction(MainWindow)
        self.actionExporter_Pixel_Art.setObjectName("actionExporter_Pixel_Art")
        self.actionQuitter = QtWidgets.QAction(MainWindow)
        self.actionQuitter.setObjectName("actionQuitter")
        self.menuFichier.addAction(self.actionNouveau)
        self.menuFichier.addAction(self.actionOuvrir)
        self.menuFichier.addAction(self.actionOuvrir_une_image)
        self.menuFichier.addSeparator()
        self.menuFichier.addAction(self.actionEnregistrer)
        self.menuFichier.addAction(self.actionEnregistrer_une_copie)
        self.menuFichier.addSeparator()
        self.menuFichier.addAction(self.actionImporter_une_Palette)
        self.menuFichier.addAction(self.actionExporter_une_Palette)
        self.menuFichier.addSeparator()
        self.menuFichier.addAction(self.actionExporter_Pixel_Art)
        self.menuFichier.addSeparator()
        self.menuFichier.addAction(self.actionQuitter)
        self.menubar.addAction(self.menuFichier.menuAction())

    def setupToolBarUi(self, MainWindow):
        # -- ToolBar Layout --
        self.scroll = QtWidgets.QScrollArea(self.centralwidget)
        self.ToolBarLayoutWidget = QtWidgets.QWidget()
        self.ToolBarLayout = QtWidgets.QVBoxLayout()

        self.scroll.setWidget(self.ToolBarLayoutWidget)
        self.ToolBarLayoutWidget.setLayout(self.ToolBarLayout)


        self.scroll.setGeometry(820, -1, 500, 551)
        self.scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.scroll.setWidgetResizable(True)

        self.ToolBarLayoutWidget.setObjectName("ToolBarLayoutWidget")

        self.ToolBarLayout.setContentsMargins(0, 0, 0, 0)
        self.ToolBarLayout.setObjectName("ToolBarLayout")
        self.ToolBarLayout.setAlignment(QtCore.Qt.AlignTop)

        self.testNewWidget = ExpandWidget()
        self.testNewWidget2 = ExpandWidget()
        self.testNewWidget3 = ExpandWidget()
        self.ToolBarLayout.addWidget(self.testNewWidget)
        self.ToolBarLayout.addWidget(self.testNewWidget2)
        self.ToolBarLayout.addWidget(self.testNewWidget3)

    # ---- 
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_image_original.setText(_translate("MainWindow", "ImageOriginals"))
        self.label_image_pixel.setText(_translate("MainWindow", "ImagePixelise"))
        #print(self.label_image_pixel.size())

        self.pushButton_ouvrir.setText(_translate("MainWindow", "Ouvir"))
        self.pushButton_generer.setText(_translate("MainWindow", "Générer"))
        self.pushButton_exporter.setText(_translate("MainWindow", "Exporter"))

        self.menuFichier.setTitle(_translate("MainWindow", "Fichier"))
        self.actionNouveau.setText(_translate("MainWindow", "Nouveau"))
        self.actionOuvrir.setText(_translate("MainWindow", "Ouvrir"))
        self.actionOuvrir_une_image.setText(_translate("MainWindow", "Ouvrir une image"))
        self.actionEnregistrer.setText(_translate("MainWindow", "Enregistrer "))
        self.actionEnregistrer_une_copie.setText(_translate("MainWindow", "Enregistrer une copie"))
        self.actionImporter_une_Palette.setText(_translate("MainWindow", "Importer une Palette"))
        self.actionExporter_une_Palette.setText(_translate("MainWindow", "Exporter une Palette"))
        self.actionExporter_Pixel_Art.setText(_translate("MainWindow", "Exporter Pixel Art"))
        self.actionQuitter.setText(_translate("MainWindow", "Quitter"))


    # ---- Action Fonction ----

    def openImage(self):
    	
        fname = QFileDialog.getOpenFileName()
        print(fname)
        print(self.label_image_original.size())
        
        self.label_image_original.setPixmap(QtGui.QPixmap(fname[0]))
        self.label_image_original.setMinimumSize(1,1)
        self.label_image_original.setPixmap(QtGui.QPixmap(fname[0]).scaled(self.label_image_pixel.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
        #---
        print("adress : ",fname[0])
        image = QtGui.QImage(fname[0])
        self.saveTemp(image)
       

# Code de Lucien permet d'afficher l'image "temp.png" qu'on updatera en faite a chaque opération sur l'image
    def generateImage(self):
        print(self.label_image_pixel.size())
        pixmap = QtGui.QPixmap("File/temp.png")
        self.label_image_pixel.setPixmap(pixmap)
        self.label_image_pixel.setMinimumSize(1,1)
        self.label_image_pixel.setPixmap(pixmap.scaled(self.label_image_pixel.size(),QtCore.Qt.KeepAspectRatio,QtCore.Qt.SmoothTransformation))
        #self.label_image_pixel.setScaledContents(True)
        
#-----------------------------------------
    def saveTemp(self,image):
    	image.save("File/temp.png")

    def exportImage(self):
        fname = QFileDialog().getSaveFileName( None, 'Save File', '', '*.png')
        print("Export Path : " , fname)
        if(fname[0] != ''):
            pixmap = QtGui.QImage("File/temp.png")
            pixmap.save(fname[0])







if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())


