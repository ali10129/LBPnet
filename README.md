# LBPnet
 Implementation of "Resource Efficient and Error Resilient Neural Networks" dissertation  
 using **pytorch** and **cuda**

 This implementation is a little bit different (might not be efficient) than what was explained in the dissertation [1], but it is *more* straightforward to understand.

I also did some small changes that you could fix to be as original as the paper suggest.




[1]: [Resource Efficient and Error Resilient Neural Networks](https://escholarship.org/uc/item/6fw3798s)



## How to use

- Please run on a GPU node!
- Make sure that `ninja` is inside your **environmnet path**:  
    - Install `ninja` from its github source and   
    add this line to your `.bash_profile` file in your **home** (~) directory
            
            export PATH=$PATH:$YourHOMEpath/ninja
                
    - Run it on Linux, not Windows! or change the `torch` header files to `at` in .cpp and .cu files.

- *module load:*

        cuda/11.4
        
        pytorch-gpu/py38/1.8  

        

- Then execute `python myLBP.py`


## License
Please feel free to use this repository.
For other purposes (e.g. commercial purposes) please contact me beforehand.

If you find this code useful in your research, please consider citing the original paper: [Resource Efficient and Error Resilient Neural Networks](https://escholarship.org/uc/item/6fw3798s)
