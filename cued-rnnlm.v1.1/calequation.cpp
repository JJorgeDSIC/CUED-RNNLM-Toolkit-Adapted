#include <stdio.h>
#include <string.h>
#include <iostream>
#include <stdlib.h>
#include <math.h>

using namespace std;
float a4 = -0.0014;
float a3 = 0.0265;
float a2 = -0.1272;
float a1 = 0.2184;
float a0 = 0.2156;

float f (float x)
{
	return (a4*pow(x, 4) + a3*pow(x,3) + a2*pow(x, 2) + a1*x + a0 );
}

float solveeqn (float y)
{
	float x;
	float x_min = 0;
	float x_max = 10;
	x = (x_max + x_max) / 2;
	if (y < 0) y = 0;
	while (fabs(f(x) - y) > 0.0001)
	{
		if (f(x) > y)
		{
			x_max = x;
			x = (x_min + x) / 2;
		}
		else
		{
			x_min = x;
			x = (x_max + x) / 2;
		}
		// printf ("x=%f, f(x)=%f, y=%f\n", x, f(x), y);
	}
	return x;
}

int main (int argc, char **argv)
{
	float y, x;
	char filename[100];
	strcpy (filename, argv[1]);
	FILE *fi = fopen (filename, "r");
	while (! feof(fi))
	{
		fscanf (fi, "%f\n", &y);
		// printf ("y=%f\n", y);
		x = solveeqn (y);
		printf ("x=%.4f\t y=%.4f\n", x, y);
	}
}

