using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;

using Emgu.CV;
using Emgu.CV.CvEnum;
using Emgu.CV.Structure;

namespace RecognizatorGosNumber_ru
{
    public partial class Form1 : Form
    {
        private NumberPlateRecognazer plateRecognazer;

        private Point startPoint;

        private Mat inputImage;
        public Form1()
        {
            InitializeComponent();
        }

        private void ProcessImage(IInputArray image)
        {
            List<IInputOutputArray> licensePlateImageList = new List<IInputOutputArray>();
            List<IInputOutputArray> filteredLicensePlateImageList = new List<IInputOutputArray>();
            List<RotatedRect> licenseboxList = new List<RotatedRect>();

            List<string> recognazedPlates = plateRecognazer.DetectedLicensePlates(image, licensePlateImageList,
                filteredLicensePlateImageList, licenseboxList);

            panel1.Controls.Clear();

            startPoint = new Point(10,10);

            for (int i=0; i<recognazedPlates.Count; i++)
            {
                Mat dest = new Mat();

                CvInvoke.VConcat(licensePlateImageList[i], filteredLicensePlateImageList[i], dest);

                AddLabelAndImage($"Номер: {recognazedPlates[i]}", dest);
            }

            Image <Bgr, byte> outputImage = inputImage.ToImage<Bgr, byte>();

            foreach (RotatedRect rect in licenseboxList)
            {
                PointF[] v = rect.GetVertices();

                PointF prevPoint = v[0];
                PointF firestPoint = prevPoint;
                PointF nextPoint = prevPoint;
                PointF lastPoint = nextPoint;

                for (int i=1; i < v.Length; i++)
                {
                    nextPoint = v[i];

                    CvInvoke.Line(outputImage, Point.Round(prevPoint), Point.Round(nextPoint), new MCvScalar(0, 0, 255), 5,
                        LineType.EightConnected, 0);

                    prevPoint = nextPoint;
                    lastPoint = prevPoint;
                }
                CvInvoke.Line(outputImage, Point.Round(lastPoint), Point.Round(firestPoint), new MCvScalar(0, 0, 255), 5,
                        LineType.EightConnected, 0);

                pictureBox1.Image = outputImage.Bitmap;
            }
        }

        private void AddLabelAndImage(string labelText, IInputArray image)
        {
            Label label = new Label();
            label.Text = labelText;
            label.Width = 100;
            label.Height = 30;
            label.Location = startPoint;

            startPoint.Y += label.Height;

            panel1.Controls.Add(label);

            PictureBox pictureBox = new PictureBox();
            Mat m = image.GetInputArray().GetMat();

            pictureBox.ClientSize = m.Size;
            pictureBox.Image = m.Bitmap;
            pictureBox.Location = startPoint;

            startPoint.Y += pictureBox.Height + 10;

            panel1.Controls.Add(pictureBox);
        }

        private void открытьToolStripMenuItem_Click(object sender, EventArgs e)
        {
            try
            {
                if(openFileDialog1.ShowDialog() == DialogResult.OK )
                {
                    pictureBox1.Image = Image.FromFile(openFileDialog1.FileName);
                }
            }
            catch (Exception ex)
            {
                MessageBox.Show(ex.Message, "Ошибка", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private void Form1_Load(object sender, EventArgs e)
        {
            plateRecognazer = new NumberPlateRecognazer(@"D:\Project\C#\RecognizatorGosNumber-ru\tessdata", "rus");
        }

        private void toolStripButton1_Click(object sender, EventArgs e)
        {
            inputImage = new Mat(openFileDialog1.FileName);

            UMat um = inputImage.GetUMat(AccessType.ReadWrite);

            ProcessImage(um);
        }
    }
}
