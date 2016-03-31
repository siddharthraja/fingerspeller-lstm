﻿using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace ImageFileConvertor
{
    class Program
    {
        static void Main(string[] args)
        {
            String[] alphabet = { "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "y", "z" };
            foreach (string ss in alphabet)
            {
                Console.WriteLine("############################# "+ss);
            
                string directoryName = "D:\\z-fingerspelling-data\\"+ss+"\\2";
                string colorDirectory = Path.Combine(directoryName, "color");
                string depthDirectory = Path.Combine(directoryName, "depth");
                string[] colorFilesList = Directory.GetFiles(colorDirectory);
                string[] depthFilesList = Directory.GetFiles(depthDirectory);
                int count = 0;
                foreach (string f in colorFilesList){
                    if (f != null && f.Length != 0)
                    {
                        if (Path.GetExtension(f).Equals(".bytes"))
                        {
                            System.Diagnostics.Debug.Write("\n" + f);
                            string filePath = colorDirectory + "\\" + ss + "_color_" + (count++) + ".png";

                            byte[] fileBytes = File.ReadAllBytes(f);
                            if (fileBytes != null && fileBytes.Length != 0) {
                                int width = 1920;
                                int height = 1080;
                                PixelFormat format = PixelFormats.Bgr32;
                                int stride = width * format.BitsPerPixel / 8;
                                BitmapFrame b = BitmapFrame.Create(BitmapSource.Create(width, height, 96, 96, format, null, fileBytes, stride));

                                using (FileStream sourceStream = new FileStream(filePath,
                                FileMode.Append, FileAccess.Write, FileShare.None,
                                bufferSize: 25000, useAsync: true))
                                {
                                    //await sourceStream.WriteAsync(encodedText, 0, encodedText.Length);
                                    BitmapEncoder encoder = new PngBitmapEncoder();
                                    //_rw.EnterReadLock();
                                    //encoder.Frames.Add(imageQueue.Dequeue());
                                    encoder.Frames.Add(b);
                                    //Thread.Sleep(100);
                                    //_rw.ExitReadLock();

                                    encoder.Save(sourceStream);
                                    sourceStream.Flush();
                                    sourceStream.Close();
                                }
                            }
                        }
                    }
                 
                }

                count = 0;
                foreach (string f in depthFilesList)
                {
                    if (f != null && f.Length != 0)
                    {
                        if (Path.GetExtension(f).Equals(".bytes"))
                        {
                            System.Diagnostics.Debug.Write("\n" + f);
                            string filePath = depthDirectory + "\\" + ss + "_depth_" + (count++) + ".png";

                            byte[] fileBytes = File.ReadAllBytes(f);
                            if (fileBytes != null && fileBytes.Length != 0) {
                                int width = 512;
                                int height = 424;
                                PixelFormat format = PixelFormats.Bgr32;
                                int stride = width * format.BitsPerPixel / 8;
                                BitmapFrame b = BitmapFrame.Create(BitmapSource.Create(width, height, 96, 96, format, null, fileBytes, stride));

                                using (FileStream sourceStream = new FileStream(filePath,
                                FileMode.Append, FileAccess.Write, FileShare.None,
                                bufferSize: 25000, useAsync: true))
                                {
                                    //await sourceStream.WriteAsync(encodedText, 0, encodedText.Length);
                                    BitmapEncoder encoder = new PngBitmapEncoder();
                                    //_rw.EnterReadLock();
                                    //encoder.Frames.Add(imageQueue.Dequeue());
                                    encoder.Frames.Add(b);
                                    //Thread.Sleep(100);
                                    //_rw.ExitReadLock();

                                    encoder.Save(sourceStream);
                                    sourceStream.Flush();
                                    sourceStream.Close();
                                }
                            }
                        }
                    }

                }

            }

        }

    }
}