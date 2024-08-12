coverage run --omit=tests/* -m pytest 
coverage report -m
coverage xml -o tests/reports/coverage.xml
coverage html --directory=tests/reports/htmlcov
genbadge coverage -i tests/reports/coverage.xml -o tests/coverage.svg
genbadge coverage -i tests/reports/coverage.xml -o docs/assets/coverage.svg