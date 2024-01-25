<h1 style="text-align: center; margin: 0;">Flask API Model Depoloyment</h1>
<h3 style="text-align: center; margin-top: 0;">Machine Learning Engineer Project</h3>
<p style="text-align: left;">The following model deployment is of a segmentation model.  It is a binary classifcation for whether a customer purchased a product (1 for purchase, 0 for no purchase) and trained on a client dataset that will not be linked here. I trained the model, saving the object and fitted imputer/scaler/feature variables, and then created an API wrapper in Flask to allow end users to receive predictions on new data by calling the the "/predict" POST route. I implemented Swagger API documentation, created unit testing with strong coverage, and endeveared to follow strong code practice. The file requirements.txt shows the required dependencies for this project.</p>
<hr>


### Project Quickstart
<p style="text-align: left;">To build the Docker image and immediately deploy the model at port 1313, run the startup bash script</p>

```bash
./run_api.txt
```

### Run unit tests, analyze coverage, and display results
```bash
./run_tests.txt
```

### API endpoints
```bash
localhost:1313/apidocs
localhost:1313/predict
```
<hr>

### Project file structure overview
```
├── api.txt
├── model_utils.txt
├── Dockerfile
├── run_api.txt
├── run_tests.txt
├── requirements.txt
├── README.md
├── write-up.pdf
├── tests
│   └── test_api.txt
├── objs
│   ├── ad_purchase_model.pickle
│   ├── variables.pickle
│   ├── imputer.pkl
│   └── std_scaler.pkl
```

<hr>

### Model processing overview
<p style="text-align: left;">To prepare input for working with our model the following steps are necessary:</p>

- Format and convert x12 and x63 fields. Common input provides x12 as a currency string in format "($xxx.xx)", so we first remove non-numerical characters and then convert the field to a float. Any of these non-numerical characters can be missing as well. Likewise, x63 is commonly provided as a percentage string in format "63.1%", so we remove the "%" character and convert it to a float as well.
- Pass the dataset through an imputer that was fitted on the training set. If any fields are missing, they are replaced with the mean value that was found  during training. We do not run the imputer on our categorical fields (x5, x31, x81, x82)
- Pass the dataset through a standard scaler that was also fitted on the training set. Once again we do not run this on our categorical fields.
- Create dummy variable encoding for our categorical fields. This will transform the day of the week field x5, for instance, into binary fields named x5_monday, x5_tuesday, etc. This will allow our model to ingest these fields.
- Subset our dataset by features that were selected during a feature selection process during training. This reduces our input size to help with speed and help reduce collinearity and the risk of overfitting. This reduces our input data to a 25-column dataset.

<br>
<hr>


### Opportunities to optimize for scalability
<p style="text-align: left;">Below are a few methods that could be used to optimize our deployment to improve scalability. At this time, none of these have been implemented in to the process</p>

<h4 style="text-align: left;">Use a different model serving framework</h4>

- I used Flask for this project because it is lightweight and easy to set up, however we would likely want to transfer to a dedicated model serving platform that would allow for better scalability, efficiency and management. It would likely be based on State Farm's vendors and licenses, but these could include TensorFlow Serving, ONNX Runtime Server, NVIDIA Triton, and more.

<h4 style="text-align: left;">Horizontal scaling</h4>

- Depending on our platform and hardware, we would be able to scale horizontally by adding more instances of our model serving as demand increases. For this we could use container orchestration with our docker image by using tools like Kubernetes or Docker Swarm. This will allow us to handle a larger load.

<h4 style="text-align: left;">Load balancing</h4>

- If we extend our deployment to run across multiple instances, it will be important to load balance these incoming requests. This ensures that the workload is evenly distributed and prevents any single instance from becoming a bottleneck.

<h4 style="text-align: left;">Improve batch processing</h4>

- Implementing concurrency with our batch process handling could speed up our throughput.

<h4 style="text-align: left;">Asynchronous Inference</h4>

- Implement asynchronous processing of requests, allowing us to handle more requests concurrently and improve throughput and responsiveness.

<h4 style="text-align: left;">Caching/Memoization</h4>

- We could implement cache strategies to reuse frequently requested results. Given the purpose of our model as a purchase classification, this would likely not be very beneficial, but is an effective general strategy.

<h4 style="text-align: left;">Autoscaling</h4>

- We could implement monitoring to observe load and dynamically horizontally scale based on usage.

<hr>

### Example single prediction query
```bash
curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' --data '{"x0": "-0.675304", "x1": "0.137379", "x2": "4.393917364", "x3": "-0.020123474", "x4": "-0.475618592", "x5": "sunday", "x6": "0.157397", "x7": "55.677997", "x8": "1.83605", "x9": "0.91846", "x10": "14.351465", "x11": "nan", "x12": "3709.93", "x13": "0.819808", "x14": "17.07728", "x15": "-0.243366", "x16": "0.061937", "x17": "14.332908999999999", "x18": "-19.662144", "x19": "0.165622", "x20": "0.146025", "x21": "-2.414621", "x22": "0.353511", "x23": "3.190204", "x24": "-118.124909", "x25": "0.90281", "x26": "0.79805", "x27": "0.5203300000000001", "x28": "14.054438000000001", "x29": "0.871179", "x30": "5.126021", "x31": "asia", "x32": "0.51033987", "x33": "2.43467728", "x34": "-2.04913613", "x35": "1.23089839", "x36": "0.83152122", "x37": "3.50526038", "x38": "-1.89375171", "x39": "-0.95390232", "x40": "-276.43", "x41": "1526.17", "x42": "-1062.4", "x43": "351.54", "x44": "0.09087572", "x45": "0.13512714", "x46": "-0.027221829", "x47": "-0.401745419", "x48": "-0.7682184759999999", "x49": "-1.477928431", "x50": "0.461940432", "x51": "1.684288945", "x52": "-0.628094413", "x53": "0.00528862", "x54": "0.38612031", "x55": "-0.80454146", "x56": "-0.215346985", "x57": "-1.265547487", "x58": "0.6828697490000001", "x59": "0.7241555059999999", "x60": "-0.11302117699999999", "x61": "-0.716963446", "x62": "-0.552213898", "x63": "45.85", "x64": "3.00265249", "x65": "4.05022364", "x66": "0.17271423", "x67": "14.03430494", "x68": "-20.88886923", "x69": "0.57667473", "x70": "0.1727856", "x71": "2.37700832", "x72": "0.48401779", "x73": "3.01276075", "x74": "-97.81706928", "x75": "nan", "x76": "1.80140824", "x77": "0.20838348", "x78": "14.4178935", "x79": "-2.58655254", "x80": "2.52245981", "x81": "November", "x82": "Male", "x83": "0.557207747", "x84": "1.7763087469999999", "x85": "0.47166523200000005", "x86": "0.789085832", "x87": "-1.061310858", "x88": "-0.850872339", "x89": "0.599991103", "x90": "-0.22179097600000003", "x91": "0.406396", "x92": "0.9239033999999999", "x93": "3.19037208", "x94": "-99.4804139", "x95": "0.65872137", "x96": "1.01721083", "x97": "0.84194747", "x98": "-32.13548212", "x99": "-92.81795904"}'
```

### Example batch prediction query
```bash
curl --request POST --url http://localhost:1313/predict --header 'content-type: application/json' --data '[{"x0": "-1.018506", "x1": "-4.180869", "x2": "5.703058724", "x3": "-0.5220215970000001", "x4": "-1.678553956", "x5": "tuesday", "x6": "0.18617", "x7": "30.162959000000004", "x8": "1.200073", "x9": "0.37312399999999996", "x10": "14.973894", "x11": "-0.81238", "x12": "$6,882.34 ", "x13": "0.078341", "x14": "32.823071999999996", "x15": "0.02048", "x16": "0.171077", "x17": "14.236199", "x18": "-18.646051", "x19": "0.575313", "x20": "0.068703", "x21": "-0.276702", "x22": "0.754378", "x23": "3.103192", "x24": "-101.889723", "x25": "1.49565", "x26": "3.4121989999999998", "x27": "0.601394", "x28": "14.210011999999999", "x29": "0.558285", "x30": "4.21066", "x31": "germany", "x32": "0.07303966", "x33": "2.99793546", "x34": "-1.91981754", "x35": "1.11327381", "x36": "-0.75988365", "x37": "3.00740356", "x38": "-1.76639977", "x39": "-1.93067723", "x40": "288.2", "x41": "129.79", "x42": "366.71", "x43": "-1134.56", "x44": "0.98441208", "x45": "1.10833973", "x46": "0.495749506", "x47": "0.42293034799999996", "x48": "1.628712455", "x49": "0.40279785799999995", "x50": "-0.272326826", "x51": "1.48269105", "x52": "-2.095101799", "x53": "0.33612654", "x54": "0.39604464", "x55": "0.43767884", "x56": "0.137700027", "x57": "0.53142961", "x58": "0.228881625", "x59": "-0.222421763", "x60": "0.561192069", "x61": "1.129407195", "x62": "0.37394123700000004", "x63": "62.59%", "x64": "33.79248734", "x65": "-0.1522697", "x66": "0.34106988", "x67": "14.39211979", "x68": "-20.60214825", "x69": "0.02168046", "x70": "0.12436805", "x71": "2.80831588", "x72": "0.48941937", "x73": "3.07847637", "x74": "-86.44286813", "x75": "0.4088527", "x76": "nan", "x77": "0.80646678", "x78": "14.02814387", "x79": "0.12779922", "x80": "3.25437849", "x81": "April", "x82": "Female", "x83": "0.460470644", "x84": "-1.1292216929999999", "x85": "-0.124149454", "x86": "-1.650432198", "x87": "-1.295166064", "x88": "0.076903248", "x89": "-1.123881898", "x90": "0.323156018", "x91": "0.04191", "x92": "0.33889244", "x93": "3.52499912", "x94": "-97.71513809999999", "x95": "1.44463704", "x96": "2.72855326", "x97": "0.71872513", "x98": "-32.94590765", "x99": "2.55535888"}, {"x0": "-1.9192200000000001", "x1": "0.451107", "x2": "4.823385218", "x3": "-2.014568798", "x4": "-1.211901352", "x5": "saturday", "x6": "0.818817", "x7": "14.490744", "x8": "4.130146", "x9": "0.314697", "x10": "14.523697", "x11": "-0.437126", "x12": "$5,647.81 ", "x13": "0.9611709999999999", "x14": "4.506393", "x15": "-0.034884", "x16": "0.377442", "x17": "14.843366", "x18": "-20.130724", "x19": "0.20491199999999998", "x20": "0.169658", "x21": "4.461883", "x22": "0.266263", "x23": "3.4797", "x24": "-89.40688399999999", "x25": "0.036239", "x26": "4.7538089999999995", "x27": "0.79404", "x28": "14.542696", "x29": "-2.001962", "x30": "5.116089", "x31": "asia", "x32": "0.29843434", "x33": "2.53769947", "x34": "-1.91607814", "x35": "0.07282569", "x36": "1.86990946", "x37": "3.01620868", "x38": "-1.02270516", "x39": "-1.62216143", "x40": "1025.23", "x41": "1638.37", "x42": "356.32", "x43": "64.98", "x44": "0.90862255", "x45": "0.07151007", "x46": "-0.264361187", "x47": "0.122219801", "x48": "-0.599600083", "x49": "-0.51763746", "x50": "-0.486018261", "x51": "-0.602458804", "x52": "-1.155273213", "x53": "0.11320627", "x54": "-0.47121462", "x55": "-0.31419697", "x56": "0.117109274", "x57": "1.433867265", "x58": "1.087831298", "x59": "1.252419764", "x60": "0.990040485", "x61": "nan", "x62": "-0.172095793", "x63": "3.11%", "x64": "18.80764932", "x65": "3.94112762", "x66": "0.23404821", "x67": "14.10636442", "x68": "-19.39177951", "x69": "0.11138828", "x70": "0.13267491", "x71": "nan", "x72": "0.07291669", "x73": "3.91079332", "x74": "-112.2446682", "x75": "nan", "x76": "1.32079944", "x77": "0.93493914", "x78": "14.02816023", "x79": "-2.0781113999999996", "x80": "5.46421613", "x81": "December", "x82": "Male", "x83": "1.270105815", "x84": "-0.298663673", "x85": "0.131659375", "x86": "0.05540591", "x87": "1.051899435", "x88": "1.270084099", "x89": "0.36856837", "x90": "1.4726412219999998", "x91": "0.385252", "x92": "0.04926468", "x93": "3.41350819", "x94": "-106.06410190000001", "x95": "0.28321709", "x96": "2.70381923", "x97": "0.7234908", "x98": "-32.16680209", "x99": "15.34720884"}, {"x0": "-0.865318", "x1": "-6.066885", "x2": "5.193225354", "x3": "-0.749214609", "x4": "-0.967170277", "x5": "thursday", "x6": "0.783379", "x7": "35.237738", "x8": "3.3919550000000003", "x9": "0.8629540000000001", "x10": "14.592314000000002", "x11": "-2.168421", "x12": "($5,032.58)", "x13": "0.669763", "x14": "45.958095", "x15": "nan", "x16": "0.422903", "x17": "14.526367", "x18": "-19.611355", "x19": "0.505279", "x20": "0.143309", "x21": "1.6974630000000002", "x22": "0.566419", "x23": "3.998022", "x24": "-100.81460600000001", "x25": "0.644827", "x26": "3.8443059999999996", "x27": "0.464636", "x28": "14.735289000000002", "x29": "-0.309609", "x30": "4.995923", "x31": "germany", "x32": "0.62960998", "x33": "2.92292401", "x34": "-1.91030666", "x35": "0.07887844", "x36": "-0.36606324", "x37": "3.94551358", "x38": "-1.59167149", "x39": "-1.5854122", "x40": "918.67", "x41": "-1050.98", "x42": "-1406.9", "x43": "-709.3", "x44": "0.9565846", "x45": "0.68484195", "x46": "-0.42424888200000005", "x47": "-0.15032657900000002", "x48": "1.748712477", "x49": "-1.053716302", "x50": "-0.851145734", "x51": "-0.470933671", "x52": "-0.498878195", "x53": "0.09582343", "x54": "-0.56534875", "x55": "1.23144852", "x56": "0.280390507", "x57": "-1.2479390129999999", "x58": "0.9426831090000001", "x59": "1.817217192", "x60": "0.452994548", "x61": "-0.9695728490000001", "x62": "-0.9914201220000001", "x63": "28.07%", "x64": "12.71192026", "x65": "-2.52049037", "x66": "0.98770888", "x67": "14.98871794", "x68": "-20.52640348", "x69": "0.69108987", "x70": "0.17633759", "x71": "-4.3015303", "x72": "0.81953402", "x73": "3.3316464999999997", "x74": "-108.77934640000001", "x75": "0.1296336", "x76": "3.84292219", "x77": "0.29238799", "x78": "14.79619203", "x79": "-0.66169388", "x80": "4.66827433", "x81": "May", "x82": "Female", "x83": "0.735642921", "x84": "-0.281013015", "x85": "-0.95694091", "x86": "0.988361594", "x87": "-1.135524825", "x88": "-0.375230455", "x89": "-0.666588535", "x90": "-1.1410206809999999", "x91": "0.560975", "x92": "0.58047134", "x93": "3.4158294999999996", "x94": "-92.08070500000001", "x95": "1.07965176", "x96": "2.55849894", "x97": "0.39826122", "x98": "-32.76053336", "x99": "-102.95087"}, {"x0": "1.4426290000000002", "x1": "1.972809", "x2": "4.015218474", "x3": "1.028498091", "x4": "0.277185087", "x5": "sunday", "x6": "0.809138", "x7": "3.217519", "x8": "3.271071", "x9": "0.708264", "x10": "14.843635999999998", "x11": "0.231055", "x12": "($1,920.03)", "x13": "0.217077", "x14": "10.224103999999999", "x15": "-3.576739", "x16": "0.722915", "x17": "14.409604000000002", "x18": "-18.685057", "x19": "0.323063", "x20": "0.24104299999999998", "x21": "4.099952", "x22": "0.114996", "x23": "3.6802099999999998", "x24": "-95.037968", "x25": "0.45015900000000003", "x26": "0.53025", "x27": "0.731528", "x28": "14.555829999999998", "x29": "-0.637542", "x30": "4.529511", "x31": "germany", "x32": "0.85863441", "x33": "2.64949973", "x34": "-1.84451307", "x35": "0.34256349", "x36": "1.19933433", "x37": "3.54231132", "x38": "-1.49496484", "x39": "-0.92297758", "x40": "1462.03", "x41": "-99.11", "x42": "-1364.83", "x43": "-1768.47", "x44": "0.2376038", "x45": "-0.21991206", "x46": "0.07602193", "x47": "1.3854386459999999", "x48": "-2.121308745", "x49": "-0.13401158", "x50": "0.829270843", "x51": "-0.493601708", "x52": "-1.128780845", "x53": "0.43678125", "x54": "-0.07830781", "x55": "-0.58221611", "x56": "0.868672654", "x57": "0.650426728", "x58": "0.9557687490000001", "x59": "-0.418021976", "x60": "1.498861402", "x61": "0.749361917", "x62": "-0.23931803", "x63": "33.49%", "x64": "39.4528548", "x65": "1.95562431", "x66": "0.42842504", "x67": "14.50193946", "x68": "-20.7736063", "x69": "0.71229497", "x70": "0.19503527", "x71": "-2.68142734", "x72": "0.9106439000000001", "x73": "3.95834052", "x74": "-103.82320109999999", "x75": "0.96465296", "x76": "1.92046507", "x77": "0.26753757", "x78": "14.10223313", "x79": "-1.01569383", "x80": "4.66821671", "x81": "December", "x82": "Female", "x83": "nan", "x84": "-0.218942704", "x85": "-0.522160524", "x86": "-1.789906337", "x87": "-2.02298006", "x88": "-0.311535357", "x89": "-1.3785222830000001", "x90": "-0.32143525", "x91": "0.064134", "x92": "0.75179874", "x93": "3.11292015", "x94": "-95.4452124", "x95": "1.07127773", "x96": "1.88649059", "x97": "0.22554158", "x98": "-32.86864401", "x99": "-20.48311936"}]'
```