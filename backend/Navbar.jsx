import styled from "styled-components";
import logo from "../assets/logo.jpeg";


function Navbar()
{

    const DIV=styled.div``;
    const UL=styled.ul`
        margin: 0;
        padding: 0;
        list-style-type: none;
        display: flex;
        justify-content: center;
        align-items: center;
        background-color: rgba(0, 0, 0, 0.17);

    `;
    const LI=styled.li`
        margin: 30px 30px 30px 30px;
       
        padding: 2px;
       
       
    `;
    const IMG=styled.img`
        display: inline;
        margin-right: 37rem;
        margin-top: 10px;
    `;
    const A=styled.a`
        text-decoration: none;
        font-family: "Montserrat", sans-serif;
        font-optical-sizing: auto;
        font-weight: 500;
        font-style: normal;
        color: rgba(0, 0, 0, 0.79);
        font-size: 21px;
    `;

    return(
        <>
            <DIV>
                <UL>
                    <IMG src={logo} height={"80px"} width={"10%"}></IMG>
                    <LI><A href={"#"}>HOME</A></LI>
                    <LI><A href={"#"}>PROBLEM</A></LI>
                    <LI><A href={"#"}>SOLUTION</A></LI>
                    <LI><A href={"#"}>IMPACT</A></LI>
                </UL>
            </DIV>
        </>
    )
}

export default Navbar;





























