# Quaternion notes
The tutorial is given by Mathoma, based on a series of youtube [videos](https://www.youtube.com/watch?v=3Ki14CsP_9k): 

## 01 Quaternions Explained Briefly

**Introduction:**

2 dimensional vector ![](http://latex.codecogs.com/gif.latex?\vec{F}=(2,1)) in ![](http://latex.codecogs.com/gif.latex?\mathbb{R}^2) space as illustrate in the below diagram: 

![](https://github.com/Erickrus/notes/blob/master/001/001.png)

Quaternion is 4 dimensional vector in ![](http://latex.codecogs.com/gif.latex?\mathbb{R}^4) space

The force can also be represented in this way
![](http://latex.codecogs.com/gif.latex?\vec{F}=(2N)\hat{x}+(1N)\hat{y})

Assume: ![](http://latex.codecogs.com/gif.latex?q=(1,2,3,4)) can be expressed the equivalent way as the 2D Force ![](http://latex.codecogs.com/gif.latex?1+2i+3j+4k)

**Addition:**

The similar as 2D vector:

![](https://github.com/Erickrus/notes/blob/master/001/002.png)

If ![](http://latex.codecogs.com/gif.latex?\vec{F}=(2,1)) and ![](http://latex.codecogs.com/gif.latex?\vec{G}=(0,1)) then ![](http://latex.codecogs.com/gif.latex?\vec{F}+\vec{G}=(2,2))

PS: From this formula, you can find the geometric meaning of the vector/quaternion addition

**Multiplication:**

The most famous equation was discovered by William Rowan Hamilton, the great Irish mathematician ![](http://latex.codecogs.com/gif.latex?i^2=j^2=k^2=ijk=-1)

![](https://github.com/Erickrus/notes/blob/master/001/004.png)

given ![](http://latex.codecogs.com/gif.latex?q_1=a+bi+cj+dk) and ![](http://latex.codecogs.com/gif.latex?q_2=e+fi+gj+hk)

![](http://latex.codecogs.com/gif.latex?q_{1}q_{2}=)

![](http://latex.codecogs.com/gif.latex?ae+afi+agj+ahk,)

![](http://latex.codecogs.com/gif.latex?bei+bfi^2+bgij+bhik,)

![](http://latex.codecogs.com/gif.latex?cej+cfji+cgj^2+chjk,)

![](http://latex.codecogs.com/gif.latex?dek+dfki+dgkj+dhk^2)

If we simplify the result using following table:

![](https://github.com/Erickrus/notes/blob/master/001/003.png)

Finally it can be further simplified in to:

![](http://latex.codecogs.com/gif.latex?ae-bf-cg-dh)

![](http://latex.codecogs.com/gif.latex?af-be+ch-dg,)

![](http://latex.codecogs.com/gif.latex?ag-bh+ce+df,)

![](http://latex.codecogs.com/gif.latex?ah+bg-cf+de)

The multiplication of quaternion is non-communicative. In general, communicative law doesn't work, we have ![](http://latex.codecogs.com/gif.latex?q_{1}q_{2}\neq%20q_{2}q_{1})

However, the association law works, e.g. 

![](http://latex.codecogs.com/gif.latex?(q_{1}q_{2})q_{3}=q_{1}(q_{2}q_{3}))

**Some jargons:**

![](http://latex.codecogs.com/gif.latex?q_1=a+bi+cj+dk=(a,b,c,d)), here:

![](http://latex.codecogs.com/gif.latex?a) is called the scalar part

![](http://latex.codecogs.com/gif.latex?(b,c,d)) is called vector part


## 02 Quaternions as 4x4 Matrices

4x4 real matrices

From previous series, we have: ![](http://latex.codecogs.com/gif.latex?q_{1}\cdot%20q_{2}=(a,b,c,d)\cdot(e,f,g,h)=)

![](http://latex.codecogs.com/gif.latex?%5Cbegin%7Bmatrix%7D%20ae-bf-cg-dh%5C%5C%20be&plus;af-dg&plus;ch%5C%5C%20ce&plus;df&plus;ag-bh%5C%5C%20de-cf&plus;bg&plus;ah%20%5Cend%7Bmatrix%7D%20%3D%20%5Cbegin%7Bbmatrix%7D%20a%20%26%20-b%20%26%20-c%20%26%20-d%20%5C%5C%20b%20%26%20a%20%26%20-d%20%26%20c%20%5C%5C%20c%20%26%20d%20%26%20a%20%26%20-b%20%5C%5C%20d%20%26%20-c%20%26%20b%20%26%20a%20%5C%5C%20%5Cend%7Bbmatrix%7D%20%5Ccdot%20%5Cbegin%7Bbmatrix%7D%20e%20%5C%5C%20f%20%5C%5C%20g%20%5C%5C%20h%20%5C%5C%20%5Cend%7Bbmatrix%7D)

This creates a link between quaternion and matrix algebra

We can think of quaternion as a matrice, where each of entries are given by the template

![](http://latex.codecogs.com/gif.latex?q_%7B1%7D%5Ccdot%20q_%7B2%7D%20%3D%20a+bi+cj+dk%3D%5Cbegin%7Bbmatrix%7D%20a%20%26%20-b%20%26%20-c%20%26%20-d%20%5C%5C%20b%20%26%20a%20%26%20-d%20%26%20c%20%5C%5C%20c%20%26%20d%20%26%20a%20%26%20-b%20%5C%5C%20d%20%26%20-c%20%26%20b%20%26%20a%20%5C%5C%20%5Cend%7Bbmatrix%7D)


Take an example:

![](http://latex.codecogs.com/gif.latex?(1,2,3,6)\cdot(0,1,0,0)=)

The template here is:

![](http://latex.codecogs.com/gif.latex?%281%2C2%2C3%2C6%29%3D%5Cbegin%7Bbmatrix%7D%201%20%26%20-2%20%26%20-3%20%26%20-6%20%5C%5C%202%20%26%201%20%26%20-6%20%26%203%20%5C%5C%203%20%26%206%20%26%201%20%26%20-2%20%5C%5C%206%20%26%20-3%20%26%202%20%26%201%20%5C%5C%20%5Cend%7Bbmatrix%7D)

![](http://latex.codecogs.com/gif.latex?%280%2C1%2C0%2C0%29%3D%5Cbegin%7Bbmatrix%7D%200%20%5C%5C%201%20%5C%5C%200%20%5C%5C%200%20%5C%5C%20%5Cend%7Bbmatrix%7D)

So the result is:

![](http://latex.codecogs.com/gif.latex?%5Cbegin%7Bbmatrix%7D%20-2%20%5C%5C%201%20%5C%5C%206%20%5C%5C%203%20%5C%5C%20%5Cend%7Bbmatrix%7D)

If we replace i, j, k with template, then we have:

![](https://github.com/Erickrus/notes/blob/master/001/005.png)

And using this, it is easy to prove: ![](http://latex.codecogs.com/gif.latex?i^2=j^2=k^2=ijk=-1)



## 03 Quaternions Extracting the Dot and Cross Products

## 04 3D Rotations and Quaternion Exponential (Special Case)

## 05 3D Rotations in General Rodrigues Rotation Formula and Quaternion Exponentials

## 06 3D Reflections with Vectors and Quaternions
