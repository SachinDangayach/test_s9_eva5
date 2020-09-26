# EVA5 Session 7 Assignment by Sachin Dangayach

**Advanced Convolutions**
**GIT Link for the package**: [https://github.com/SachinDangayach/EVA5](https://github.com/SachinDangayach/EVA5)

**Collab Link**: https://colab.research.google.com/drive/1S0wf6mBJGQhYpkI7p38leuicX_5Mto2u?usp=sharing

**A. Target**

1.  Change the code such that it uses GPU
2.  Change the architecture to C1C2C3C40 (basically 3 MPs)
3.  Total RF must be more than 44
4.  One of the layers must use Depthwise Separable Convolution
5.  One of the layers must use Dilated Convolution
6.  Use GAP (compulsory):- add FC after GAP to target #of classes (optional)
7.  Achieve 80% accuracy, as many epochs as you want. Total Params to be less than 1M.

**B. Results**

1.  Parameters: 815,200
2.  Best Training Accuracy in 20 epochs: 87.72%
3.  Best Test Accuracy in 20 epochs: 85.09 %
4.  Total RF reached: 76*76 at the end of Conv block 4

**C. Analysis**

Model meets all the mentioned targets. I have used the Depthwise seprable con and dilated kernel in the conv block 3. Also, As train test gap was coming more in earlier versions, I have used the horizontal flip and random rotations to act as regularizer and reduce the train test gap. Code is modularized and uploaded as package in git repo.
