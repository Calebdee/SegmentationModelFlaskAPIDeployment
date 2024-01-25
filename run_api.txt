docker build -t ad_purchase_app .
docker run -p 1313:1313 ad_purchase_app
echo "Model running on port 1313, see docs at http://localhost:1313/apidocs"