import convert_data
import train_image_classifier_gen
import export_inference_graph
import freeze_graph
import evaluation
import os
import json
import time

d_a = {}

with open('default_arguments.json', 'r') as js:
	d_a = json.load(js)

def train(dt):
	if not 'dataset_dir' in dt:
		return "No dataset directory specified"
	for key in dt:
		d_a[key] = dt[key]
	
	d_a['freeze_dict']['input_checkpoint'] = os.path.join(d_a['train_dir'], 'model.ckpt-'+ str(d_a['max_number_of_steps'])) 
	
	start = time.time()
	train_size, val_size, num_classes = convert_data.run(dt['dataset_dir'], d_a['dataset_name'], 
																	d_a['validation_percentage'], d_a['num_shards'])
	
	start_train = time.time()
	train_image_classifier_gen.main(d_a['train_dir'], d_a['num_clones'], d_a['clone_on_cpu'], 
       train_size, val_size, num_classes, d_a['worker_replicas'], 
       d_a['log_every_n_steps'], d_a['save_interval_secs'], d_a['weight_decay'],
       d_a['optimization'], d_a['learning_rate'], d_a['moving_average_decay'], 
       d_a['dataset'], d_a['max_number_of_steps'], d_a['checkpoint'])
	end_train = time.time()

	export_inference_graph.main(d_a['export'], train_size, val_size, num_classes)
	
	freeze_graph.main(d_a['freeze_dict'])
	end = time.time()

	train_time = end_train - start_train
	full_time = end - start
	rest = full_time - train_time

	return { 'training_time': train_time,
					 'full_time': full_time,
					 'rest': rest
				 }

def inference(image):
	return evaluation.main(image)
