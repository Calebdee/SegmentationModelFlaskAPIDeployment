echo "(y/n) For unit testing, we will convert .txt python files into .py. Do you wish to continue?"
read answer

if [ "$answer" != "${answer#[Yy]}" ] ;then 
    if [ ! -e "api.py" ]; then
        cp "api.txt" "api.py"
    fi
    if [ ! -e "model_utils.py" ]; then
        cp "model_utils.txt" "model_utils.py"
    fi
    if [ ! -e "tests/test_api.py" ]; then
        cp "tests/test_api.txt" "tests/test_api.py"
    fi

    coverage run -m unittest discover -s "tests/"
    coverage report
    coverage html
    cd htmlcov/
    echo "View code coverage visually at localhost:8088"
    python -m http.server 8088
else
    echo "Received input that wasn't y/Y, will not run unit tests"
fi

