function downloadFile(){
    src = "TemplatePDF.pdf"
    var link=document.createElement('a');
    document.body.appendChild(link);
    link.href= src;
    link.download = '';
    link.click();
}

// input.setAttribute('size', input.getAttribute('placeholder').length);
function letterOnly(input){
    var regex = /[^A-Ea-e]/gi
    input.value = input.value.replace(regex, "")
}

function insertAlternativas(){

    var quantAlt = document.getElementById('quantAlt').value;
    if (! quantAlt == ""){

        var divAlternativas = document.getElementById("alternativas");
        divAlternativas.innerHTML = ""
        content = ""
        
        for (var i = 1; i <= quantAlt; i++) {

            if (i < 10){
                pos = "0" + String(i);
            }
            else{
                pos = String(i);
            }

            content = content + "<div class = 'alternativa'>"
            content = content + "<label for='"+ pos +"'>"+ pos +"</label>"
            content = content + "<input type='text' id='"+ pos +"' name='alt"+ pos +"' placeholder='A B C D E' maxlength='1' onkeyup='this.value = this.value.toUpperCase();letterOnly(this);'></input>"
            content = content + "</div>"
         }
         divAlternativas.innerHTML = content

    }   
}