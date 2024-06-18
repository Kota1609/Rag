 1372  docker ps -a
 1373  history | grep docker
 1374  docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1375  docker ps -a
 1376  docker exec -it ollama /bin/bash
 1470  docker ps -a
 1471  docker exec -it ollama /bin/bash
 1472  docker stop ollama
 1473  docker rmi ollama/ollama 
 1474  docker rm d2539529e24d
 1475  docker ps -a
 1476  docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1477  cat /etc/docker/daemon.json
 1478  sudo systemctl restart docker
 1480  systemctl restart docker
 1481  systemctl status docker
 1531  docker run -d --gpus '"device=0,2,3"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1532  docker ps -a
 1533  history | grep docker
 1534  docker exec -it ollama /bin/bash
 1535  docker run -d --gpus '"device=0,2,3"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1536  docker ps -a
 1537  docker exec -it ollama /bin/bash
 1538  docker rm 2c2d699fa8a8
 1539  docker run -d --gpus '"device=0,2,3"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1540  docker ps -a
 1541  docker rm 4ccca908ab78
 1542  docker run -d --gpus '"device=1"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1543  docker rmi ollama/ollama
 1544  docker rmi 4ccca908ab78
 1545  docker ps -a
 1546  docker rmi 9e77215d3960
 1547  docker rmiollama/ollama
 1548  docker rmi ollama/ollama
 1549  docker ps -a
 1550  docker rmi 9e77215d3960
 1551  docker rmi 65276675bdd5
 1552  docker rmi 9e77215d3960
 1553  docker rmi ollama
 1554  docker rmi -f ollama/ollama
 1555  docker ps -a
 1556  docker rmi -f 65276675bdd5
 1557  docker rmi -f 9e77215d3960
 1558  docker rm 9e77215d3960
 1559  docker run -d --gpus '"device=1"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1563  history | grep docker
 1564  docker ps -a
 1565  docker run -d --gpus '"device=1"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1566  docker exec -it ollama /bin/bash
 1567  docker rm 72dbb3e0e769
 1568  docker ps -a
 1569  docker run -d --gpus '"device=1"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1571  docker run -d --gpus '"device=1"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1572  docker ps -a
 1573  docker rm 727bfd9b36a6
 1574  docker run -d --gpus '"device=1"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1640  docker run -d -p 3000:8080 -e OLLAMA_API_BASE_URL=http://myserver.com:11434 --name ollama-webui --restart always ghcr.io/ollama-webui/ollama-webui:main
 1642  docker ps -a
 1645  docker ps -a
 1646  docker rm 072c346ad099
 1647  docker rm 423d406735f5
 1648  docker rm 072c346ad099
 1649  docker rmi 072c346ad099
 1650  docker ps -a
 1651  docker rmi ghcr.io/ollama-webui/ollama-webui:main
 1652  docker rm ghcr.io/ollama-webui/ollama-webui:main
 1653  history | grep docker
 1654  docker run -d --gpus '"device=1"' -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1665  docker exec -it ollama /bin/bash
 1666  docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1667  docker ps -a
 1668  docker rm 1910b785c8fc
 1669  docker stop ollama-webui
 1670  docker rm ollama-webui
 1671  docker rmi ghcr.io/ollama-webui/ollama-webui:main
 1672  docker ps -a
 1673  docker rmi 1910b785c8fc
 1674  docker rm 1910b785c8fc
 1675  docker stop ollama
 1676  docker rm ollama
 1677  docker rmi ollama/ollama
 1678  docker ps -a
 1679  docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1680  docker exec -it ollama ollama run llama3
 1681  docker exec -it ollama ollama list
 1682  docker exec -it ollama ollama run gemma
 1683  docker exec -it ollama ollama list
 1696  docker exec -it ollama ollama run llama3:70b-instruct-fp16
 1697  docker exec -it ollama ollama list
 1699  docker exec -it ollama ollama rm wizardlm2:8x22b
 1700  docker exec -it ollama ollama rm llama3:70b-instruct-q8_0
 1701  docker exec -it ollama ollama rm mixtral:8x22b-instruct
 1702  docker exec -it ollama ollama rm mixtral:latest
 1703  docker exec -it ollama ollama rm llama3:70b
 1704  docker exec -it ollama ollama list
 1705  docker exec -it ollama ollama rm mixtral:8x22b
 1708  docker exec -it ollama ollama run llama3:70b-instruct-fp16
 1709  docker exec -it ollama ollama list
 1723  history | grep docker
 1737  docker exec -it ollama ollama run llama3:70b-instruct-fp16
 1738  docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1739  docker ps -a
 1740  docker rm 72f649dfea4a
 1741  docker rm ollama
 1742  docker stop ollama
 1743  docker rm ollama
 1744  docker rmi ollama/ollama
 1745  docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 1746  docker ps -a
 1747  docker exec -it ollama ollama list
 1748  docker exec -it ollama ollama run llama3:70b-instruct-fp16
 1749  docker exec -it ollama
 1750  docker exec -it ollama/ollama
 1751  history | grep docker
 1753  docker exec -it ollama /bin/bash
 1754  docker ps -a
 1755  docker stop ollama
 1756  docker rm ollama
 1761  docker exec -it ollama ollama list
 1773  history | grep docker
 1774  docker exec -it ollama ollama list
 1803  docker ps -a
 1804  docker run -d --gpus=all -v ollama:/root/.ollama -p 11434:11434 --name ollama ollama/ollama
 2002  history | grep docker
