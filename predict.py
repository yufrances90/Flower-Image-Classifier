import sys
import click

import utils.helper as uh

@click.group()
def main():
    pass

@main.command('predict')
@click.option('--image-filepath', help='The filepath of an image', required=True)
@click.option('--model-filepath', default='./model.h5', help='The filepath of a saved Keras model')
def predict_class(**kwargs):

    '''
    Reads in an image and a saved Keras model & 
    Prints the most likely image class and it's associated probability
    '''

    model_filepath = kwargs['model_filepath']  

    model = uh.load_keras_model(model_filepath)

    image_filepath = kwargs['image_filepath'] 

    probs, classes = uh.predict(image_filepath, model)

    click.echo(probs)
    click.echo(classes)

if __name__ == '__main__':
    main()