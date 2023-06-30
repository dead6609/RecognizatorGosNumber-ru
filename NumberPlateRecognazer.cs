using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Drawing;

using Emgu;
using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.OCR;
using Emgu.CV.Structure;
using Emgu.CV.Util;
using Emgu.Util;

namespace RecognizatorGosNumber_ru
{
    internal class NumberPlateRecognazer : DisposableObject
    {
        private Tesseract ocr;

        public NumberPlateRecognazer(string tessdataPath, string lang)
        {
            ocr = new Tesseract(tessdataPath, lang, OcrEngineMode.TesseractLstmCombined);
        }

        public List<string> DetectedLicensePlates(IInputArray image,
            List<IInputOutputArray> licensePlateImageList,
            List<IInputOutputArray> fillteredLicensePlateImageList,
            List<RotatedRect> detectedLicensePlateRegionList)
        {
            List<string> licenses = new List<string>();

            using (Mat gray = new Mat())
            {
                using (Mat canny = new Mat())
                {
                    using (VectorOfVectorOfPoint contours = new VectorOfVectorOfPoint())
                    {
                        CvInvoke.CvtColor(image, gray, ColorConversion.Bgr2Gray);

                        CvInvoke.Canny(gray, canny, 100, 50, 3, false);

                        int[,] hierachy = CvInvoke.FindContourTree(canny, contours, ChainApproxMethod.ChainApproxSimple);

                        FindLicensPlate(contours, hierachy, 0, gray, canny, 
                            licensePlateImageList, fillteredLicensePlateImageList,
                            detectedLicensePlateRegionList, licenses);
                    }
                }
            }
            return licenses;
        }

        //Список с госзнаками в тестовом представлении и прямоугольники с ними
        private void FindLicensPlate(VectorOfVectorOfPoint contours, int[,] hierachy, 
            int index, 
            IInputArray gray, IInputArray canny,
            List<IInputOutputArray> licensePlateImageList,
            List<IInputOutputArray> filteredLicensePlateImageList,
            List<RotatedRect> detectedLicensePlateRegionList,
            List<string> license)
        {
            for (; index >=0; index = hierachy[index, 0])
            {
                int numderOfChildren = GetNumberOfChildren(hierachy, index);

                if (numderOfChildren == 0)
                    continue;

                using (VectorOfPoint contour = contours[index])
                {
                    if ( CvInvoke.ContourArea(contour) > 400)
                    {
                        if (numderOfChildren < 3)
                        {
                            FindLicensPlate(contours, hierachy, hierachy[index, 2], gray, canny,
                                licensePlateImageList, filteredLicensePlateImageList, detectedLicensePlateRegionList,
                                license);

                            continue;
                        }

                        RotatedRect box = CvInvoke.MinAreaRect(contour);

                        if (box.Angle < -45.0)
                        {
                            float tmp = box.Size.Width;

                            box.Size.Width = box.Size.Height;
                            box.Size.Height = tmp;

                            box.Angle += 90.0f;
                        }
                        else if(box.Angle >45.0)
                        {
                            float tmp = box.Size.Width;

                            box.Size.Width = box.Size.Height;
                            box.Size.Height = tmp;

                            box.Angle -= 90.0f;
                        }

                        double whRatio = (double)box.Size.Width / box.Size.Height;

                        if (!(3.0 < whRatio && whRatio < 10))
                        {
                            if (hierachy[index, 2] > 0)
                            {
                                FindLicensPlate(contours, hierachy, hierachy[index, 2], gray, canny,
                                    licensePlateImageList, filteredLicensePlateImageList, detectedLicensePlateRegionList,
                                    license);

                                continue;
                            }
                        }

                        using (UMat tmp1 = new UMat())
                        {
                            using (UMat tmp2 = new UMat())
                            {
                                PointF[] srcCorners = box.GetVertices();

                                PointF[] destCorners = new PointF[]
                                {
                                    new PointF(0, box.Size.Height -1),
                                    new PointF(0,0),
                                    new PointF(box.Size.Width -1,0),
                                    new PointF(box.Size.Width -1, box.Size.Height -1)
                                };

                                using (Mat rot = CvInvoke.GetAffineTransform(srcCorners, destCorners))
                                {
                                    CvInvoke.WarpAffine(gray, tmp1, rot, Size.Round(box.Size));
                                }

                                Size approxSize = new Size(240, 100);

                                double scale = Math.Min(approxSize.Width / box.Size.Width, approxSize.Height / box.Size.Height);

                                Size newSize = new Size((int)Math.Round(box.Size.Width * scale), (int)Math.Round(box.Size.Height *scale));

                                CvInvoke.Resize(tmp1, tmp2 , newSize , 0, 0, Inter.Cubic);

                                int edgePixelSize = 3;

                                Rectangle newRoi = new Rectangle(new Point(edgePixelSize, edgePixelSize), 
                                    tmp2.Size - new Size(2*edgePixelSize, 2* edgePixelSize));

                                UMat plate = new UMat(tmp2, newRoi);

                                UMat filteredPlate = FilterPlate(plate);

                                StringBuilder stringBuilder = new StringBuilder();

                                using (UMat tmp = filteredPlate.Clone())
                                {
                                    ocr.SetImage(tmp);

                                    ocr.Recognize();

                                    stringBuilder.Append(ocr.GetUTF8Text());
                                }

                                license.Add(stringBuilder.ToString());

                                licensePlateImageList.Add(plate);

                                filteredLicensePlateImageList.Add(filteredPlate);

                                detectedLicensePlateRegionList.Add(box);
                            }
                        }
                    }
                }
            }
        }

        private int GetNumberOfChildren(int[,] hierarchy, int index)
        {
            index = hierarchy[index, 2];
            if (index < 0)
                return 0;
            int count = 1;
            while (hierarchy[index, 0] > 0)
            {
                count++;
                index = hierarchy[index, 0];
            }
            return count;
        }

        private static UMat FilterPlate(UMat plate)
        {
            UMat thresh = new UMat();

            CvInvoke.Threshold(plate, thresh, 120, 255, ThresholdType.BinaryInv);

            Size plateSize = plate.Size;

            using (Mat plateMask = new Mat(plateSize.Height, plateSize.Width, DepthType.Cv8U, 1))
            {
                using (Mat plateCanny = new Mat())
                {
                    using (VectorOfVectorOfPoint contourus = new VectorOfVectorOfPoint())
                    {
                        plateMask.SetTo(new MCvScalar(255.0));

                        CvInvoke.Canny(plate, plateCanny, 100, 50);

                        CvInvoke.FindContours(plateCanny, contourus, null, RetrType.External, ChainApproxMethod.ChainApproxSimple);

                        int count = contourus.Size;

                        for (int i = 0; i < count; i++)
                        {
                            using(VectorOfPoint contour = contourus[i])
                            {
                                Rectangle rect = CvInvoke.BoundingRectangle(contour);

                                if (rect.Height > (plateSize.Height >> 1))
                                {
                                    rect.X -= 1;
                                    rect.Y -= 1;

                                    rect.Width += 2;
                                    rect.Height += 2;

                                    Rectangle roi =new Rectangle(Point.Empty, plate.Size);

                                    rect.Intersect(roi);

                                    CvInvoke.Rectangle(plateMask, rect, new MCvScalar(), -1);


                                }
                            }
                        }

                        thresh.SetTo(new MCvScalar(), plateMask);
                    }
                }
            }

            CvInvoke.Erode(thresh, thresh, null, new Point(-1, -1), 1, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);

            CvInvoke.Dilate(thresh, thresh, null, new Point(-1, -1), 1, BorderType.Constant, CvInvoke.MorphologyDefaultBorderValue);

            return thresh;
        }

        protected override void DisposeObject()
        {
            ocr.Dispose();
        }
    }
}
