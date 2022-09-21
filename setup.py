import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
     name='beret-beige',  
     version='0.0.2',
     author="Jayoung Ryu",
     author_email="jayoung_ryu@g.harvard.edu",
     description="Bayesian Effect size Inference with Guide counts and Editing rate",
     long_description=long_description,
     long_description_content_type="text/markdown",
     url="https://github.com/jykr/beige",
     packages=setuptools.find_packages(),
     scripts=["bin/beige"],
     install_requires=[
        'pyro-ppl',
        'numpy',
        'berets'
      ],
      include_package_data=True,
    #package_data={'': ['beret/annotate/ldlr_exons.fa']},
     classifiers=[
         "Programming Language :: Python :: 3",
         "Operating System :: OS Independent",
     ],
 )

