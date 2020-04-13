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

## 03 Quaternions Extracting the Dot and Cross Products

## 04 3D Rotations and Quaternion Exponential (Special Case)

## 05 3D Rotations in General Rodrigues Rotation Formula and Quaternion Exponentials

## 06 3D Reflections with Vectors and Quaternions
