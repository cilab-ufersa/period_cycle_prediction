import setuptools

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name='period_cycle_prediction',
    url='https://github.com/cilab-ufersa/period_cycle_prediction',
    author='CILAB',
    author_email='rosana.rego@ufersa.edu.br',
    # Needed to actually package something
    packages=setuptools.find_packages(),
    include_package_data=True,
    # Needed for dependencies
    install_requires=required,
    description='A package to predict the period cycle',
    long_description=open('README.md').read(),
)