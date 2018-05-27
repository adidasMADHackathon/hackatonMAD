# Penalty forecast app | Repo 1/2
## Team Centaury 

[Repo 2](https://github.com/soyjuanmacias/centaury-app-react-native)

Run backend:
`docker run -p 5000:5000 gfelixc/hackaton`

Dataset:
`http://localhost:5000/dataset`

Forecast:
`curl -X POST http://localhost:5000/forecast -H 'Cache-Control: no-cache' -H 'Content-Type: application/json' -d '{"anger": [0.0360],"contempt": [0.1651],"disgust": [0.1309],"fear": [0.0271],"happiness": [0.0286],"neutral": [0.0804],"sadness": [0.0183],"surprise": [0.5136]}'`