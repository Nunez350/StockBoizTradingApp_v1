import React from 'react';
import '../App.css';
import ReactDOM from'react-dom'


function Contact(){
    const contactPageConst = (
      <div id="contactPage">
          <h1 id="contactPageHeading">Contact us!</h1>
          <p id="contactPageParagraph">Please take all complaints to Roy Nunez!</p>
      </div>
    );
    ReactDOM.render(
      contactPageConst,
      document.getElementById('root')
    );
    return contactPageConst;
}

export default Contact;