# https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/deployment-setup-configuration
# https://docs.microsoft.com/en-us/azure/machine-learning/desktop-workbench/model-management-service-deploy

# install Azure ML Workbench (or just the CLI, if available separately)
# setup the accounts and environments (first link)
# run the following commands to test locally

az ml model --model [path] --name [model name]
az ml manifest create --manifest-name [manifest name] -f score.py -r python -c conda.yml -s schema.json
az ml image create -n [image name] --manifest-id [the manifest ID]
az ml service create realtime --image-id [image id] -n [service name]