import sys
import click

import utils.helper as uh

@click.group()
def main():
    pass

@main.command('predict')
@click.option('--image-filepath', help='The filepath of an image', required=True)
@click.option('--model-filepath', default='./model.h5', help='The filepath of a saved Keras model')
@click.option(
    '--json-filepath', 
    default='./label_map.json', 
    help='JSON file that maps the class values to other category names'
)
@click.option(
    '--top-k', 
    default=1, 
    help='the top K classes along with associated probabilities'
)
def predict_class(**kwargs):

    '''
    Reads in an image and a saved Keras model & 
    Prints the most likely image class and it's associated probability
    '''

    model_filepath = kwargs['model_filepath']  

    model = uh.load_keras_model(model_filepath)

    image_filepath = kwargs['image_filepath'] 

    json_filepath = kwargs['json_filepath']

    top_k = kwargs['top_k']

    probs, labels = uh.predict(image_filepath, model, json_filepath, top_k)

    click.echo('****************************************************************************')
    click.echo('**********************      Final Result    ********************************')

    for index in range(top_k):

        msg = '{}. {} with probability {}'.format(index+1, labels[index], probs[index])
        
        click.echo(msg)

    click.echo('****************************************************************************')

if __name__ == '__main__':
    main()