#include<iostream>  
#include<string>  
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>








using namespace cv;
using namespace std;
void task(string name);



float getBlur_Laplacian_sp(const Mat& inMat) //简易模糊度
{
	Mat greyMat, dst;
	cv::cvtColor(inMat, greyMat, CV_BGR2GRAY);//转换为灰度图

	Laplacian(greyMat, dst, inMat.depth());

	Scalar mean, stddev;
	meanStdDev(dst, mean, stddev);//计算方差
	return  10*(1/(stddev.val[0]));

}


void getBlur_Laplacian_sp_print(const Mat& inMat) //稳定清晰度
{
	Mat greyMat, dst;
	cv::cvtColor(inMat, greyMat, CV_BGR2GRAY);//转换为灰度图

	Laplacian(greyMat, dst, inMat.depth());

	Scalar mean, stddev;
	meanStdDev(dst, mean, stddev);//平均

	cout << "clarity:" << mean.val[0] << "\n"
		<< "blurriness:" << 10 * (1 / (stddev.val[0])) << "\n";
}


double getPSNR(const Mat& I1, const Mat& I2)
{
	Mat s1;
	absdiff(I1, I2, s1);       // |I1 - I2|
	s1.convertTo(s1, CV_32F);  // cannot make a square on 8 bits
	s1 = s1.mul(s1);           // |I1 - I2|^2

	Scalar s = sum(s1);         // sum elements per channel

	double sse = s.val[0] + s.val[1] + s.val[2]; // sum channels

	if (sse <= 1e-10) // 对于小值返回零
		return 0;
	else
	{
		double  mse = sse / (double)(I1.channels() * I1.total());
		double psnr = 10.0*log10((255 * 255) / mse);
		return psnr;
	}
}






Scalar getMSSIM(const Mat& i1, const Mat& i2)
{
	const double C1 = 6.5025, C2 = 58.5225;
	/***************************** INITS **********************************/
	int d = CV_32F;

	Mat I1, I2;
	i1.convertTo(I1, d);           // cannot calculate on one byte large values
	i2.convertTo(I2, d);

	Mat I2_2 = I2.mul(I2);        // I2^2
	Mat I1_2 = I1.mul(I1);        // I1^2
	Mat I1_I2 = I1.mul(I2);        // I1 * I2

								   /*************************** END INITS **********************************/

	Mat mu1, mu2;   // PRELIMINARY COMPUTING
	GaussianBlur(I1, mu1, Size(11, 11), 1.5);
	GaussianBlur(I2, mu2, Size(11, 11), 1.5);

	Mat mu1_2 = mu1.mul(mu1);
	Mat mu2_2 = mu2.mul(mu2);
	Mat mu1_mu2 = mu1.mul(mu2);

	Mat sigma1_2, sigma2_2, sigma12;

	GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
	sigma1_2 -= mu1_2;

	GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
	sigma2_2 -= mu2_2;

	GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
	sigma12 -= mu1_mu2;

	///////////////////////////////// FORMULA ////////////////////////////////
	Mat t1, t2, t3;

	t1 = 2 * mu1_mu2 + C1;
	t2 = 2 * sigma12 + C2;
	t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

	t1 = mu1_2 + mu2_2 + C1;
	t2 = sigma1_2 + sigma2_2 + C2;
	t1 = t1.mul(t2);               // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

	Mat ssim_map;
	divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

	Scalar mssim = mean(ssim_map); // mssim = average of ssim map
	return mssim;
}






float getMSSIM_mono(const Mat& i1, const Mat& i2)
{
	Mat i1_g, i2_g;
	cv::cvtColor(i1, i1_g, CV_BGR2GRAY);//转换为灰度图
	cv::cvtColor(i2, i2_g, CV_BGR2GRAY);//转换为灰度图

	float f = getMSSIM(i1_g, i2_g).val[0];
	return f;
}





float getMSSIM_c(const Mat& i1, const Mat& i2,int c)
{
	Mat i1_g[3], i2_g[3];

	Mat imageSplit[3];
	split(i1, i1_g);
	split(i2, i2_g);

	float f = getMSSIM(i1_g[c], i2_g[c]).val[0];
	return f;
}

float getMSSIM_quik(const Mat& i1, const Mat& i2)
{
	using namespace cv;
	Mat i1_g, i2_g;
	cv:cvtColor(i1, i1_g, CV_BGR2GRAY);//转换为灰度图
	cv::cvtColor(i2, i2_g, CV_BGR2GRAY);//转换为灰度图


	Mat i1_d(400, 400, i1_g.type());
	Mat i2_d(400, 400, i2_g.type());

	if (i1_g.cols*i1_g.rows > 400 * 400)
	{
		resize(i1_g, i1_d, i1_d.size(), 0, 0, INTER_NEAREST);
		resize(i2_g, i2_d, i2_d.size(), 0, 0, INTER_NEAREST);
	}

	float f = getMSSIM(i1_d, i2_d).val[0];
	return f;
}






//-------

double Entropy(Mat img)
{
	double temp[256] = { 0.0 };

	// 计算每个像素的累积值
	for (int m = 0; m<img.rows; m++)
	{// 有效访问行列的方式
		const uchar* t = img.ptr<uchar>(m);
		for (int n = 0; n<img.cols; n++)
		{
			int i = t[n];
			temp[i] = temp[i] + 1;
		}
	}

	// 计算每个像素的概率
	for (int i = 0; i<256; i++)
	{
		temp[i] = temp[i] / (img.rows*img.cols);
	}

	double result = 0;
	// 计算图像信息熵
	for (int i = 0; i<256; i++)
	{
		if (temp[i] == 0.0)
			result = result;
		else
			result = result - temp[i] * (log(temp[i]) / log(2.0));
	}

	return result;
}



void task(string name)
{
	Mat img = imread(name);
	Mat org = imread("Q1.jpg");


	if (!img.data)
	{
		cout << "open image file failed.";//载入失败
		return;
	}


	cout << name << ":" << endl;
	getBlur_Laplacian_sp_print(img);
	cout << "getPSNR: " << getPSNR(org, img) << endl;

	cout << "Entropy: " << Entropy( img) << endl;
	


	double time;
	
	time = (double)getTickCount();
	cout << "getMSSIM_Q: " << getMSSIM_quik(org, img) << endl;
	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
	cout << "[ " << time << " ]"<<endl;


	time = (double)getTickCount();
	cout << "getMSSIM_MONO: " << getMSSIM_mono(org, img) << endl;
	time = 1000 * ((double)getTickCount() - time) / getTickFrequency();
	cout << "[ " << time << " ]" << endl;




}


//---------------------------------


int main(int argc, char* argv[])
{
	
	//for (int a = 0; a < argc; a++)
	//{
	//	std::cout << a << "/" << argc << ":" << argv[a] << "\n";
	//}

	if (argc == 2)
	{
		cout << argv[1] << ":" << endl;
		Mat img = imread(argv[1]);

		if (!img.data)
		{
			cout << "open image file failed.";//载入失败
			return -1;
		}

		getBlur_Laplacian_sp_print(img);
		cout << "entropy:" << Entropy(img) << "\n";
	}
	if (argc ==3)
	{
		cout << argv[1] << ":" << endl;
		Mat img = imread(argv[1]);
		Mat org = imread(argv[2]);

		if (!img.data|| !org.data)
		{
			cout << "open image file failed.";//载入失败
			return -1;
		}

		getBlur_Laplacian_sp_print(img);
		cout << "entropy:" << Entropy(img) << "\n";
		cout << "PSNR:" << getPSNR(org, img) << "\n";
		cout << "SSIM:" << getMSSIM_mono(img, org) << "\n";
	}

	if (argc == 4)
	{
		cout << argv[1] << ":" << endl;
		Mat img = imread(argv[1]);
		Mat org = imread(argv[2]);

		if (!img.data || !org.data)
		{
			cout << "open image file failed.";//载入失败
			return -1;
		}

		getBlur_Laplacian_sp_print(img);
		cout << "entropy:" << Entropy(img) << "\n";
		cout << "PSNR:" << getPSNR(org, img) << "\n";

		if ( String(argv[3]) == "-q")
		{
			cout << "SSIM:" << getMSSIM_quik(img, org) << "\n";
		}
		if (String(argv[3]) == "-a")
		{
			cout << "SSIM:" << getMSSIM(img, org) << "\n";
		}
	}

	cout << endl;







	//task("1.png");
	//task("1_B.png");
	//task("2.jpg");
	//task("3.jpg");
	//task("4.jpg");
	//task("5.jpg");
	//namedWindow("原图");
	//imshow("原图", img);

	//task("Q1.jpg");
	//task("Q_WEB3.jpg");
	//task("Q_ppduck.jpg");
	//task("Q_tinyjpg.jpg");
	//task("Q1_50.jpg");
	//task("Q1_30.jpg");
	//task("Q1_x.jpg");
	////task("W1.PNG");
	////task("W2.PNG");

	//waitKey();
	//getchar();

	return 0;
}



