from setuptools import setup

setup(
    # Needed to silence warnings (and to be a worthwhile package)
    name='period_cycle_prediction',
    url='https://github.com/cilab-ufersa/period_cycle_prediction',
    author='Rosana Rego',
    author_email='rosana.rego@ufersa.edu.br',
    # Needed to actually package something
    packages=['utils', 'notebooks'],
    # Needed for dependencies
    install_requires=['numpy','pandas', 'pendulum', 'matplotlib'],
    # *strongly* suggested for sharing
    version='0.1',
    # The license can be anything you like
    license='MIT',
    description='A package to predict the period cycle',
    # We will also need a readme eventually (there will be a warning)
    # long_description=open('README.txt').read(),
)