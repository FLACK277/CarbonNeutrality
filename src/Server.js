const express=require("express");
const PORT=process.env.PORT||3000;
const App=express();

App.get("/index.html",(req,res)=>{
    res.send("Hello from Server");
})

App.listen(PORT, () => {
    console.log(`Express server running at http://localhost:${PORT}/`);
});





















