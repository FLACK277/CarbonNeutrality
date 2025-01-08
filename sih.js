const http=require('http');
function index(request,response){
  response.writeHead(200);// write head is for status code
  response.end('hi');
  console.log("hello from server")
}
http.createServer(function(request,response){
  if(request.url==='sih2.html')
  {
    return index(request,response);
  }
  response.writeHead(404);
  response.end(http.STATUS_CODES[404]);
}).listen(3000);