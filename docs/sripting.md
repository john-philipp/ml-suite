# Scripting

Any workflow can be scripted. A simple workflow is as below:

```
actions:
    - model build-sample-data
    - model prepare --seq-end -1
    - model train -e1
    - model test -p -v
```

You can run this via (stdin):
```
mls scripts run --from-stdin <<EOF
actions:
    - model build-sample-data
    - model prepare --seq-end -1
    - model train -e1
    - model test -p -v
EOF
```

Alternatively store the script content as a file.

## Jinja2 templating

Within each script you have Jinja2 available.
```
mls scripts run --from-stdin <<EOF
actions:
    # Build and predict sample data.
    - model build-sample-data
    - model prepare --seq-end -1
    
    # Start training.
    - model train -e1
    
    # Train into latest existing training.
    {% for x in range(2): %}
    - model train -e1 -i
    {% endfor %}
    
    # Test, build, and show predictions.
    - model test -p -v
EOF
```

This allows us to train in cycles. This isn't *that* interesting right now. Because we're training on the same data. And we can just increase the number of epochs. However, once we start training on multiple datasets, the resourcefulness of this approach should become clearer.

```
mls scripts run --from-stdin <<EOF
actions:
    # Build two datasets.
    - model build-sample-data # Dataset 1.
    - model build-sample-data # Dataset 2.
    
    # Prepare the latest two datasets.
    - model prepare -d-1 --seq-end -1
    - model prepare -d-2 --seq-end -1
    
    # Train once on dataset 1.
    # Creates training record.
    - model train -d-2 -e1
    
    # Train once on dataset 2.
    # Into latest existing training.
    - model train -d-1 -e1 -i
    
    # Train based on the two datasets.
    # But alternate, avoiding overfitting.
    {% for x in range(5): %}
    - model train -d-2 -e1 -i
    - model train -d-1 -e1 -i
    {% endfor %}
    
    # Test, build, and show predictions.
    - model test -p -v
EOF
```

## Bindings
Bindings allow us to specify and vary certain parameters at runtime.