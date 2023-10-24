raw_file=`pip show transformers | grep Location | awk '{print substr($0, 11)}'`
cp ./utils.py '/'$raw_file'/transformers/generation/'
