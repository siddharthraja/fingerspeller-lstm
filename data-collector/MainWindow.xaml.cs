﻿//---------------------------------------------------------------------------------------------------
// <copyright file="MainWindow.xaml.cs" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
// <Description>
// This program tracks up to 6 people simultaneously.
// If a person is tracked, the associated gesture detector will determine if that person is seated or not.
// If any of the 6 positions are not in use, the corresponding gesture detector(s) will be paused
// and the 'Not Tracked' image will be displayed in the UI.
// </Description>
//----------------------------------------------------------------------------------------------------

namespace Microsoft.Samples.Kinect.FingerSpellingData
{
    using System;
    using System.Collections.Generic;
    using System.Collections.ObjectModel;
    using System.ComponentModel;
    using System.Windows;
    using System.Windows.Input;
    using System.Windows.Controls;
    using Microsoft.Kinect;
    using System.Windows.Media;
    using System.Windows.Shapes;
    using Microsoft.Kinect.VisualGestureBuilder;
    using System.Windows.Forms;
    using System.Windows.Media.Imaging;
    using System.Globalization;
    using System.IO;
    using System.Resources;
   //#using Microsoft.Xna.Framework;
    //using SlimDX;
    /// <summary>
    /// Interaction logic for the MainWindow
    /// </summary>
    public partial class MainWindow : Window, INotifyPropertyChanged
    {
        /// <summary> Active Kinect sensor </summary>
        private KinectSensor kinectSensor = null;
        
        /// <summary> Array for the bodies (Kinect will track up to 6 people simultaneously) </summary>
        private Body[] bodies = null;

        /// <summary> Reader for body frames </summary>
        private BodyFrameReader bodyFrameReader = null;

        /// <summary> Current status text to display </summary>
        private string statusText = null;

        /// <summary> KinectBodyView object which handles drawing the Kinect bodies to a View box in the UI </summary>
        private KinectBodyView kinectBodyView = null;
        
        /// <summary> List of gesture detectors, there will be one detector created for each potential body (max of 6) </summary>
        private List<GestureDetector> gestureDetectorList = null;

        //private ClientInterface clientInterface = null;

        private bool startMode = false;
        bool raisedLeftHand = false;

        private ColorFrameReader colorFrameReader = null;
        private WriteableBitmap colorBitmap = null;
        //############# PHRASE NAME ########################### PHRASE NAME ########################## PHRASE NAME ########################################
        private String phrase_name = "Alligator in wagon";
        /// <summary>
        /// Initializes a new instance of the MainWindow class
        /// </summary>
        public MainWindow()
        {
            // only one sensor is currently supported
            this.kinectSensor = KinectSensor.GetDefault();
            
            // set IsAvailableChanged event notifier
            this.kinectSensor.IsAvailableChanged += this.Sensor_IsAvailableChanged;

            // open the sensor
            this.kinectSensor.Open();

            // set the status text
            this.StatusText = this.kinectSensor.IsAvailable ? Properties.Resources.RunningStatusText
                                                            : Properties.Resources.NoSensorStatusText;

            // open the reader for the body frames
            this.bodyFrameReader = this.kinectSensor.BodyFrameSource.OpenReader();

            // set the BodyFramedArrived event notifier
            this.bodyFrameReader.FrameArrived += this.Reader_BodyFrameArrived;

            // initialize the BodyViewer object for displaying tracked bodies in the UI
            this.kinectBodyView = new KinectBodyView(this.kinectSensor);

            // initialize the gesture detection objects for our gestures
            this.gestureDetectorList = new List<GestureDetector>();

            // initialize the MainWindow
            this.InitializeComponent();

            // set our data context objects for display in UI
            this.DataContext = this;
            this.kinectBodyViewbox.DataContext = this.kinectBodyView;

            // open the reader for the color frames
            this.colorFrameReader = this.kinectSensor.ColorFrameSource.OpenReader();

            // wire handler for frame arrival
            this.colorFrameReader.FrameArrived += this.Reader_ColorFrameArrived;

            // create the colorFrameDescription from the ColorFrameSource using Bgra format
            FrameDescription colorFrameDescription = this.kinectSensor.ColorFrameSource.CreateFrameDescription(ColorImageFormat.Bgra);

            // create the bitmap to display
            this.colorBitmap = new WriteableBitmap(colorFrameDescription.Width, colorFrameDescription.Height, 96.0, 96.0, PixelFormats.Bgr32, null);
            /*
            // connect to htk server via tcpClient
            this.clientInterface = ClientInterface.getClientInstance();
            clientInterface.connect();

            Console.WriteLine("connect to the client interface \n " + clientInterface.GetHashCode() + "\n");            
            //clientInterface.disconnect();*/

            // create a gesture detector for each body (6 bodies => 6 detectors) and create content controls to display results in the UI
            int col0Row = 0;
            int col1Row = 0;
            int maxBodies = this.kinectSensor.BodyFrameSource.BodyCount;
            for (int i = 0; i < maxBodies; ++i)
            {
                GestureResultView result = new GestureResultView(i, false, false, 0.0f);
                GestureDetector detector = new GestureDetector(this.kinectSensor, result);
                this.gestureDetectorList.Add(detector);                
                
                // split gesture results across the first two columns of the content grid
                ContentControl contentControl = new ContentControl();
                contentControl.Content = this.gestureDetectorList[i].GestureResultView;
                /*
                if (i % 2 == 0)
                {
                    // Gesture results for bodies: 0, 2, 4
                    Grid.SetColumn(contentControl, 0);
                    Grid.SetRow(contentControl, col0Row);
                    ++col0Row;
                }
                else
                {
                    // Gesture results for bodies: 1, 3, 5
                    Grid.SetColumn(contentControl, 1);
                    Grid.SetRow(contentControl, col1Row);
                    ++col1Row;
                }

                this.contentGrid.Children.Add(contentControl);*/
            }

            prevDeleteButton.Click += deletePreviousSample;
            saveRgb.Click += ScreenshotButton_Click;
            String imagepath = "../../../alphabet_images/"+"y"+".png";
            imageHolder.Background = new ImageBrush(new BitmapImage(new Uri(@imagepath, UriKind.RelativeOrAbsolute)));

        }

        /// <summary>
        /// INotifyPropertyChangedPropertyChanged event to allow window controls to bind to changeable data
        /// </summary>
        public event PropertyChangedEventHandler PropertyChanged;

        /// <summary>
        /// Gets or sets the current status text to display
        /// </summary>
        public string StatusText
        {
            get
            {
                return this.statusText;
            }

            set
            {
                if (this.statusText != value)
                {
                    this.statusText = value;

                    // notify any bound elements that the text has changed
                    if (this.PropertyChanged != null)
                    {
                        this.PropertyChanged(this, new PropertyChangedEventArgs("StatusText"));
                    }
                }
            }
        }

        /// <summary>
        /// Execute shutdown tasks
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void MainWindow_Closing(object sender, CancelEventArgs e)
        {
            if (this.bodyFrameReader != null)
            {
                // BodyFrameReader is IDisposable
                this.bodyFrameReader.FrameArrived -= this.Reader_BodyFrameArrived;
                this.bodyFrameReader.Dispose();
                this.bodyFrameReader = null;
            }

            if (this.gestureDetectorList != null)
            {
                // The GestureDetector contains disposable members (VisualGestureBuilderFrameSource and VisualGestureBuilderFrameReader)
                foreach (GestureDetector detector in this.gestureDetectorList)
                {
                    detector.Dispose();
                }

                this.gestureDetectorList.Clear();
                this.gestureDetectorList = null;
            }

            if (this.kinectSensor != null)
            {
                this.kinectSensor.IsAvailableChanged -= this.Sensor_IsAvailableChanged;
                this.kinectSensor.Close();
                this.kinectSensor = null;
            }
        }

        /// <summary>
        /// Handles the event when the sensor becomes unavailable (e.g. paused, closed, unplugged).
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void Sensor_IsAvailableChanged(object sender, IsAvailableChangedEventArgs e)
        {
            // on failure, set the status text
            this.StatusText = this.kinectSensor.IsAvailable ? Properties.Resources.RunningStatusText
                                                            : Properties.Resources.SensorNotAvailableStatusText;
        }

        /// <summary>
        /// Handles the color frame data arriving from the sensor
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void Reader_ColorFrameArrived(object sender, ColorFrameArrivedEventArgs e)
        {
            // ColorFrame is IDisposable
            using (ColorFrame colorFrame = e.FrameReference.AcquireFrame())
            {
                if (colorFrame != null)
                {
                    FrameDescription colorFrameDescription = colorFrame.FrameDescription;

                    using (KinectBuffer colorBuffer = colorFrame.LockRawImageBuffer())
                    {
                        this.colorBitmap.Lock();

                        // verify data and write the new color frame data to the display bitmap
                        if ((colorFrameDescription.Width == this.colorBitmap.PixelWidth) && (colorFrameDescription.Height == this.colorBitmap.PixelHeight))
                        {
                            colorFrame.CopyConvertedFrameDataToIntPtr(
                                this.colorBitmap.BackBuffer,
                                (uint)(colorFrameDescription.Width * colorFrameDescription.Height * 4),
                                ColorImageFormat.Bgra);

                            this.colorBitmap.AddDirtyRect(new Int32Rect(0, 0, this.colorBitmap.PixelWidth, this.colorBitmap.PixelHeight));
                        }

                        this.colorBitmap.Unlock();
                    }

                    if (startMode)
                    {
                        SaveRGBScreenshot();
                    }
                }
            }
        }

        private void ScreenshotButton_Click(object sender, RoutedEventArgs e)
        {
            SaveRGBScreenshot();
        }

        private void SaveRGBScreenshot()
        {
            if (this.colorBitmap != null)
            {
                // create a png bitmap encoder which knows how to save a .png file
                BitmapEncoder encoder = new PngBitmapEncoder();

                // create frame from the writable bitmap and add to encoder
                encoder.Frames.Add(BitmapFrame.Create(this.colorBitmap));

                string time = System.DateTime.Now.ToString("hh'-'mm'-'ss", CultureInfo.CurrentUICulture.DateTimeFormat);

                string myPhotos = Environment.GetFolderPath(Environment.SpecialFolder.MyDocuments);
                myPhotos = myPhotos + "\\FingerSpellingData";
                Console.WriteLine(myPhotos);
                string path = System.IO.Path.Combine(myPhotos, "KinectScreenshot-Color-" + time + ".png");

                // write the new file to disk
                try
                {
                    // FileStream is IDisposable
                    using (FileStream fs = new FileStream(path, FileMode.Create))
                    {
                        encoder.Save(fs);
                    }

                    this.StatusText = "saved to " + myPhotos;//string.Format(Properties.Resources.SavedScreenshotStatusTextFormat, path);
                }
                catch (IOException)
                {
                    this.StatusText = string.Format(Properties.Resources.FailedScreenshotStatusTextFormat, path);
                }
            }
        }

        /// <summary>
        /// Handles the body frame data arriving from the sensor and updates the associated gesture detector object for each body
        /// </summary>
        /// <param name="sender">object sending the event</param>
        /// <param name="e">event arguments</param>
        private void Reader_BodyFrameArrived(object sender, BodyFrameArrivedEventArgs e)
        {
            bool dataReceived = false;

            using (BodyFrame bodyFrame = e.FrameReference.AcquireFrame())
            {
                if (bodyFrame != null)
                {
                    if (this.bodies == null)
                    {
                        // creates an array of 6 bodies, which is the max number of bodies that Kinect can track simultaneously
                        this.bodies = new Body[bodyFrame.BodyCount];
                    }

                    // The first time GetAndRefreshBodyData is called, Kinect will allocate each Body in the array.
                    // As long as those body objects are not disposed and not set to null in the array,
                    // those body objects will be re-used.
                    bodyFrame.GetAndRefreshBodyData(this.bodies);
                    dataReceived = true;
                }
            }

            if (this.bodies != null)
            {
                int maxBodies = this.kinectSensor.BodyFrameSource.BodyCount;
                for (int i = 0; i < maxBodies; ++i)
                {
                    Body body = this.bodies[i];
                    SolidColorBrush gSolidColor = new SolidColorBrush();
                    gSolidColor.Color = Color.FromRgb(0, 255, 0);
                    SolidColorBrush rSolidColor = new SolidColorBrush();
                    rSolidColor.Color = Color.FromRgb(255, 0, 0);

                    Joint handr = body.Joints[JointType.HandRight];         //11
                    Joint handl = body.Joints[JointType.HandLeft];          //7
                    Joint thumbr = body.Joints[JointType.ThumbRight];       //24
                    Joint thumbl = body.Joints[JointType.ThumbLeft];        //22
                    Joint tipr = body.Joints[JointType.HandTipRight];       //23
                    Joint tipl = body.Joints[JointType.HandTipLeft];        //21

                    Joint hipr = body.Joints[JointType.HipRight];           //16
                    Joint hipl = body.Joints[JointType.HipLeft];            //12
                    Joint spinebase = body.Joints[JointType.SpineBase];     //0
                    Joint spinemid = body.Joints[JointType.SpineMid];

                    double spineDifferenceY = Math.Abs(spinebase.Position.Y - spinemid.Position.Y);
                    double distFromBase = (spineDifferenceY * 2.0) / 3.0; //Take 2/3rds the distance from the spine base.
                    double threshold = spinebase.Position.Y + distFromBase;

                    double handlY = handl.Position.Y;
                    double handrY = handr.Position.Y;

                    if (threshold > handrY)
                    {
                        rectangleFlag.Fill = rSolidColor;
                        if (textFlag.Text == "Stopped.")
                        {
                            //First time, when beginning
                            textFlag.Text = "Ready!";
                        }
                        else if (textFlag.Text == "Started!")
                        {
                            if (!raisedLeftHand)
                            {
                                //Erase the session data
                                //clientInterface.sendData("delete");
                                startMode = false;
                                textFlag.Text = "Erased data, and ready!";
                                raisedLeftHand = false;
                            }
                            else if (raisedLeftHand)
                            {
                                //Save the session data
                                startMode = false;
                                //clientInterface.sendData("end");
                                Console.WriteLine("\nEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEEE\n");
                                textFlag.Text = "Saved data, and ready!";
                                raisedLeftHand = false;
                            }

                        }
                    }
                    else if (threshold < handrY && textFlag.Text != "Stopped.")
                    {
                        //Begin the data collection.
                        if (!startMode)
                        {
                            textFlag.Text = "Started!";
                            startMode = true;
                            //clientInterface.sendData("start");
                            //clientInterface.sendData(phrase_name);
                            Console.WriteLine("\nSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSSS\n");
                        }
                        rectangleFlag.Fill = gSolidColor;
                        textFlag.Text = "Started!";
                        if (threshold < handlY)
                        {
                            raisedLeftHand = true;
                        }
                    }
                }
            }

            if (dataReceived)
            {
                // visualize the new body data
                this.kinectBodyView.UpdateBodyFrame(this.bodies);

                // we may have lost/acquired bodies, so update the corresponding gesture detectors
                if (this.bodies != null)
                {
                    // loop through all bodies to see if any of the gesture detectors need to be updated
                    int maxBodies = this.kinectSensor.BodyFrameSource.BodyCount;
                    for (int i = 0; i < maxBodies; ++i)
                    {
                        Body body = this.bodies[i];
                        ulong trackingId = body.TrackingId;
                        
                        if (trackingId != 0)
                        {
                            
                            String msg = prepareTcpMessage(body);                            
                            //clientInterface.sendData(msg);

                        }
                           
                        // if the current body TrackingId changed, update the corresponding gesture detector with the new value
                        if (trackingId != this.gestureDetectorList[i].TrackingId)
                        {
                            this.gestureDetectorList[i].TrackingId = trackingId;
                            
                            // if the current body is tracked, unpause its detector to get VisualGestureBuilderFrameArrived events
                            // if the current body is not tracked, pause its detector so we don't waste resources trying to get invalid gesture results
                            this.gestureDetectorList[i].IsPaused = trackingId == 0;
                        }
                    }
                }
            }
        }


        private String checkForHandLocation(Body body)
        {            
            return "";
        }

        private static int msgCount = 0;
        private String prepareTcpMessage(Body body)
        {
            String msg = "";
            
            Joint head = body.Joints[JointType.Head];               //3
            Joint neck = body.Joints[JointType.Neck];               //2
            Joint shoulderr = body.Joints[JointType.ShoulderRight]; //8
            Joint shoulderl = body.Joints[JointType.ShoulderLeft];  //4
            Joint spinesh = body.Joints[JointType.SpineShoulder];   //20

            Joint elbowr = body.Joints[JointType.ElbowRight];       //9
            Joint elbowl = body.Joints[JointType.ElbowLeft];        //5
            Joint wristr = body.Joints[JointType.WristRight];       //10
            Joint wristl = body.Joints[JointType.WristLeft];        //6
            Joint handr = body.Joints[JointType.HandRight];         //11
            Joint handl = body.Joints[JointType.HandLeft];          //7
            Joint thumbr = body.Joints[JointType.ThumbRight];       //24
            Joint thumbl = body.Joints[JointType.ThumbLeft];        //22
            Joint tipr = body.Joints[JointType.HandTipRight];       //23
            Joint tipl = body.Joints[JointType.HandTipLeft];        //21

            Joint hipr = body.Joints[JointType.HipRight];           //16
            Joint hipl = body.Joints[JointType.HipLeft];            //12
            Joint spinebase = body.Joints[JointType.SpineBase];     //0
            Joint kneer = body.Joints[JointType.KneeRight];         //17
            Joint kneel = body.Joints[JointType.KneeLeft];          //13
            
            double l0 = Math.Round(Math.Sqrt(Math.Pow((neck.Position.X - shoulderl.Position.X), 2) + Math.Pow((neck.Position.Y - shoulderl.Position.Y), 2) + Math.Pow((neck.Position.Z - shoulderl.Position.Z), 2)), 5);
            double r0 = Math.Round(Math.Sqrt(Math.Pow((neck.Position.X - shoulderr.Position.X), 2) + Math.Pow((neck.Position.Y - shoulderr.Position.Y), 2) + Math.Pow((neck.Position.Z - shoulderr.Position.Z), 2)), 5);
            double l1 = Math.Round(Math.Sqrt(Math.Pow((shoulderl.Position.X - elbowl.Position.X), 2) + Math.Pow((shoulderl.Position.Y - elbowl.Position.Y), 2) + Math.Pow((shoulderl.Position.Z - elbowl.Position.Z), 2)), 5);
            double r1 = Math.Round(Math.Sqrt(Math.Pow((shoulderr.Position.X - elbowr.Position.X), 2) + Math.Pow((shoulderr.Position.Y - elbowr.Position.Y), 2) + Math.Pow((shoulderr.Position.Z - elbowr.Position.Z), 2)), 5);
            double l2 = Math.Round(Math.Sqrt(Math.Pow((elbowl.Position.X - wristl.Position.X), 2) + Math.Pow((elbowl.Position.Y - wristl.Position.Y), 2) + Math.Pow((elbowl.Position.Z - wristl.Position.Z), 2)), 4);
            double r2 = Math.Round(Math.Sqrt(Math.Pow((elbowr.Position.X - wristr.Position.X), 2) + Math.Pow((elbowr.Position.Y - wristr.Position.Y), 2) + Math.Pow((elbowr.Position.Z - wristr.Position.Z), 2)), 4);

            double norm = (l0 + l1 + l2 + r0 + r1 + r2) / 2.0;

            Joint[] joints = { head, neck, shoulderr, shoulderl, spinesh, elbowr, elbowl, wristr, wristl, handr, handl, thumbr, thumbl, tipr, tipl, hipr, hipl, spinebase, kneer, kneel };
            String msg_points = "";
            foreach(Joint j in joints){
                msg_points += "" + Math.Round(j.Position.X, 5) + " " + Math.Round(j.Position.Y, 5) + " " + Math.Round(j.Position.Z, 5) + " ";
            }
            Console.WriteLine(msgCount++ +" | " + msg.Length);

            JointType[] joint_types = {JointType.Head, JointType.Neck, JointType.ShoulderRight, JointType.ShoulderLeft, JointType.SpineShoulder, JointType.ElbowRight, JointType.ElbowLeft, JointType.WristRight, JointType.WristLeft, JointType.HandRight, JointType.HandLeft, JointType.ThumbRight, JointType.ThumbLeft, JointType.HandTipRight, JointType.HandTipLeft, JointType.HipRight, JointType.HipLeft, JointType.SpineBase };//, JointType.KneeRight, JointType.KneeLeft };
            int joint_count = 0;
            foreach (JointType j in joint_types)
            {
                Microsoft.Kinect.Vector4 quat = body.JointOrientations[j].Orientation;
                double msg_w = Math.Round( quat.W, 7 );
                double msg_x = Math.Round( quat.X, 7 );
                double msg_y = Math.Round( quat.Y, 7 );
                double msg_z = Math.Round( quat.Z, 7 );
                
                msg += "" + msg_w + " " + msg_x + " " + msg_y + " " + msg_z + " ";
                joint_count++;
            }

            msg = msg + " ||| " + msg_points;
            return msg;
        }

        private float my_clamp(float val, float min, float max)
        {
            if (val.CompareTo(min) < 0) return min;
            else if (val.CompareTo(max) > 0) return max;
            else return val;
        }

        private void deletePreviousSample(object sender, RoutedEventArgs e)
        {
            //clientInterface.sendData("delete");
            startMode = false;
            textFlag.Text = "Erased previous sample, and ready!";
        }

    }

}

