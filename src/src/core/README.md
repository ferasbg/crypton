# CORE
- Have more control over the dataset itself before extending the features to partitions etc. Make sure that it can first work on a defined dataset and the tuple in the same script rather than working with more than enough complexity involved with the problem in the first place.
- NSL requires particular data formatting into a dict, but that creates errors with the CifarClient, unless I merge the tuple and access an iterable np.ndarray with xy_test[0] for image set and xy_test[1] for target label set.
- I can also resolve returned values by emulating the history/results dict that is returned for .eval inside a flwr.client sub-class.
- I can reduce complexity by having control over customizing how every process is executed. To an extent, working with other sample code isn't the best way to solve the problem, and especially when all the methods have been exhausted and a re-implementation with more control is a better option.
- server is not under any form of adv. regularization regardless of its initial client state.
- currently I need to setup the federated client with respect to adv_reg
- first configure the adv_reg client model to client
- configure client and server
- configure custom parse_args to modify important variables
- add metrics
- currently solving the error of callable param of label; some simple errors with processing mnist dataset
- it's very much do-able to have adv_reg clients
- adding .perturb_bath during server-side eval is needed
- formalization of robustness of entire system is different than metrics collected to assess exp. configs
- solve input layer compatability error
- solve dataset processing error with adv_reg with adv_model with Client



## Unused Code

```

perturbed_images, labels, predictions = [], [], []

for batch in test_set_for_adv_model:
  perturbed_batch = reference_model.perturb_on_batch(batch)
  # Clipping makes perturbed examples have the same range as regular ones.
  perturbed_batch[IMAGE_INPUT_NAME] = tf.clip_by_value(                          
      perturbed_batch[IMAGE_INPUT_NAME], 0.0, 1.0)
  y_true = perturbed_batch.pop(LABEL_INPUT_NAME)
  perturbed_images.append(perturbed_batch[IMAGE_INPUT_NAME].numpy())
  labels.append(y_true.numpy())
  predictions.append({})
  for name, model in models_to_eval.items():
    y_pred = model(perturbed_batch)
    metrics[name](y_true, y_pred)
    predictions[-1][name] = tf.argmax(y_pred, axis=-1).numpy()

for name, metric in metrics.items():
  print('%s model accuracy: %f' % (name, metric.result().numpy()))


  
perturbed_images, labels, predictions = [], [], []

# We want to record the accuracy.
metric = tf.keras.metrics.SparseCategoricalAccuracy()

for batch in test_set_for_adv_model:
  # Record the loss calculation to get the gradient.
  with tf.GradientTape() as tape:
    tape.watch(batch)
    losses = labeled_loss_fn(batch[LABEL_INPUT_NAME],
                             base_model(batch[IMAGE_INPUT_NAME]))
    
  # Generate the adversarial example.
  fgsm_images, _ = nsl.lib.adversarial_neighbor.gen_adv_neighbor(
      batch[IMAGE_INPUT_NAME],
      losses,
      fgsm_nbr_config,
      gradient_tape=tape
  )

  # Update our accuracy metric.
  y_true = batch['label']
  y_pred = base_model(fgsm_images)
  metric(y_true, y_pred)

  # Store images for later use.
  perturbed_images.append(fgsm_images)
  labels.append(y_true.numpy())
  predictions.append(tf.argmax(y_pred, axis=-1).numpy())

print('%s model accuracy: %f' % ('base', metric.result().numpy()))



perturbed_images, labels, predictions = [], [], []

# Record the accuracy.
metric = tf.keras.metrics.SparseCategoricalAccuracy()

for batch in test_set_for_adv_model:
  # Gradient tape to calculate the loss on the first iteration.
  with tf.GradientTape() as tape:
    tape.watch(batch)
    losses = labeled_loss_fn(batch[LABEL_INPUT_NAME],
                             base_model(batch[IMAGE_INPUT_NAME]))
    
  # Generate the adversarial examples.
  pgd_images, _ = nsl.lib.adversarial_neighbor.gen_adv_neighbor(
      batch[IMAGE_INPUT_NAME],
      losses,
      pgd_nbr_config,
      gradient_tape=tape,
      pgd_model_fn=pgd_model_fn,
      pgd_loss_fn=pgd_loss_fn,
      pgd_labels=batch[LABEL_INPUT_NAME],
  )

  # Update our accuracy metric.
  y_true = batch['label']
  y_pred = base_model(pgd_images)
  metric(y_true, y_pred)

  # Store images for visualization.
  perturbed_images.append(pgd_images)
  labels.append(y_true.numpy())
  predictions.append(tf.argmax(y_pred, axis=-1).numpy())

print('%s model accuracy: %f' % ('base', metric.result().numpy()))



if (adv_reg_m2):
    # method 2
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    train_dataset = tf.data.Dataset.from_tensor_slices({'image': x_train, 'label': tf.convert_to_tensor(y_train, dtype='float32')}).batch(batch_size=32)
    test_dataset = tf.data.Dataset.from_tensor_slices({'image': x_test, 'label': tf.convert_to_tensor(y_test, dtype='float32')}).batch(batch_size=32)



```
