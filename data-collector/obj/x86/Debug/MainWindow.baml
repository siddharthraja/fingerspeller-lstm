   M S B A M L   `   `   `  ����I  Edata-collector, Version=1.0.0.0, Culture=neutral, PublicKeyToken=nulli9clr-namespace:Microsoft.Samples.Kinect.FingerSpellingData+Microsoft.Samples.Kinect.FingerSpellingData  5   	   <    6Microsoft.Samples.Kinect.FingerSpellingData.MainWindow   5   	   W SPresentationCore, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35R NWindowsBase, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35\ XPresentationFramework, Version=4.0.0.0, Culture=neutral, PublicKeyToken=31bf3856ad364e35D 9http://schemas.microsoft.com/winfx/2006/xaml/presentation    5   	   R NSystem.Xaml, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b77a5c561934e089:x,http://schemas.microsoft.com/winfx/2006/xaml     5   	   Klocal9clr-namespace:Microsoft.Samples.Kinect.FingerSpellingData     5   	   -   5          � Title$$  Finger Spelling Data Collector��5   	   $	��650��5   	   $	��750��6    � 	Resources 5         MediumGreyBrush �� Color  KinectPurpleBrush  KinectBlueBrush �� DataTypeC   =Microsoft.Samples.Kinect.FingerSpellingData.GestureResultView � RowDefinitions �� StringFormat��%�  &	        &	 #     &	 F     (�� i     5      *, +)�� 5	   
   $ 	#ff6e6e6e��62   �� 5
   
   $ 	#ff52318f��64   �� 5   
   $ 	#ff00BCF2��62   �� 5   
   "  6   .R�5      � $
��Auto��6   $��5q�6    ��6+   �� *	BodyColor+ 5      �� 5      $
��Auto �6$   �� 5      �� 5      $
��Auto �6$   �� 5      $
��Auto �6$   .+�5      �� $��Stretch=�6   $��14(�6u   $��Segoe UI)�6�   $��SemiBold"�6�   $��5q�6�   ��6:   �� *	BodyIndex+$ Body Index: {0}��6�   �� 5      $��Uniform=�67   $��1��6I   $��5q�6V   ��6   �� *ImageSource+6V   �� 5      $��18(�6R   $��Segoe UI)�6`   $��SemiBold"�6v   $��2��6�   $��5 5 0 0q�6�   ��6   �� *
Detected+$ Seated: {0}��6�   �� 5      $��18(�6X   $��Segoe UI)�6f   $��SemiBold"�6|   $��3��6�   $��5 0 0 0q�6�   ��6   �� *
Confidence+$ Confidence: {0}��6�   5      5      5      .��5      � $��	10 0 10 0q�6    5   
   �� 5       $
��Auto �6   �� 5!      $��* �6   �� 5"      $
��Auto �6   .+�5$   
   �� $��0��6   $��	0,0,301,5q�6!   $��Right=�64   $��Bottom=�6P   #��[  6k   $
��Bold"�6�   $��Segoe UI)�6�   $��18(�6�   $��Sign Catcher��6�   $	��114��6�    J� RenderTransformOrigin$ -0.604,0.506&�6�   �� 5%   
   $��0��6   $��Images\Status.png��6   $
��None=�68   $��Center=�6G   $��
662,0,10,5q�6d   $ 1.776,0.356&�6x   $��50��6�   � 5'   
   -   5         �Name$ contentGrid��5'      $��1��6$   $ 
0.33,0.522&�61   $��0,5,0,0q�6T    � ColumnDefinitions 5(      �� 5)      $��1* �6#   �� 5*      $��1* �6#   �� 5+      $��2* �6#    5-      �� 5.      �� 5/      �� 50      .+�52      �� -   5        $��simpleLabel1��52   �   $��RECORDING STATUS��6   	 �� TextAlignment$	 Center=�60   $
��Wrap=�6G   $��	0,73,24,0q�6[   #��[  6n   $
��Bold"�6�   $��Segoe UI)�6�   $��14(�6�   $ ��40                        ��6�   $	��Top=�6
  $��Right=�6"  $	��188��6>  $��2��6J  $��2��6Z  �� 53      -   5        $��rectangleFlag��53   �   $��2��6   $��44,73,217,10q�6(   $��Center=�6>   $��Center=�6[   $	��102��6v   $	��100��6�   
���  ��$��2��6�   �� 54      -   5        $��textFlag��54   �   $��Stopped.��6   $	 Center=�6(   $
��Wrap=�6?   $��
191,0,0,10q�6S   #��[  6g   $
��Bold"�6�   $��Segoe UI)�6�   $��14(�6�   $��2��6�   $
��Left=�6�   $	��100��6  $��71��6  $��Bottom=�6  $��2��66  �� 55      -   5        $��prevDeleteButton��55   L   $��Delete Last��6   $
��Bold"�6,   $��16(�6>   $��Right=�6d   $	��100��6�   $��75��6�   $	��Top=�6�   $��	0,10,71,0q�6�   $��2��6�   �� 56      -   5        $��imageHolder��56   A   $�� ��6   $
��Bold"�6!   $��16(�63   $��Right=�6T   $	��341��6p   $	��348��6|   $	��Top=�6�   $��	0,10,10,0q�6�   $��2��6�   $��2��6�   �� 57      -   5        $��saveRgb��57   K   $��
Save Frame��6   $
��Bold"�6+   $��16(�6=   $��Right=�6Z   $	��100��6v   $��75��6�   $	��Top=�6�   $��	0,10,70,0q�6�   $��2��6�   $��1��6�   �� 5=      
 �� Orientation$
 Vertical=�6   $��2��60   $��3��6@   $��
20 10 0 10q�6Q   $��Center=�6e   $��Center=�6�   .+�5>      (� -	   5         (�Name$ kinectBodyViewbox��5>      .��5?      �� $��UniformToFill=�6;   ��6   �� *ImageSource+6;   5@      �� 5A      $6��0Stand/Sit in front of the sensor to get tracked.��6   $
��Wrap=�6T   $��5 10 5 5q�6h   #��[  6z   $��Segoe UI)�6�   $��14(�6�   �� 5B      -
   5        $
��sign��5B      $�� ��6(   $
��Wrap=�60   $��5 10 5 5q�6D   #��[  6V   $��Segoe UI)�6�   $��16(�6�   5C      5D      �� 5F   
   -   5        $��	statusBar��5F   ?   $��2��6   $��Stretch=�6!   $��Bottom=�6P   
�������#��[  6~   .@�5G      �� ��6   �� *
StatusText+5H      5I      5J      