insertAlternativas(30);
insertOutras(10);

function downloadFile(){
    src = "TemplatePDF.pdf"
    var link=document.createElement('a');
    document.body.appendChild(link);
    link.href= src;
    link.download = '';
    link.click();
}

function numberOnly(t){
    t.value = t.value.replace(/[^0-9]/g,'');
}

function letterOnly(input){
    var regex = /[^A-Ea-e]/gi
    input.value = input.value.replace(regex, "")
}

function insertAlternativas(quantAlt){

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

        content = content + "<div class = 'resposta'>"
        content = content + "<label for='alt"+ pos +"'>"+ pos +"</label>"
        content = content + "<input type='text' id='alt"+ pos +"' name='alt"+ pos +"' placeholder='A B C D E' maxlength='1' onkeyup='this.value = this.value.toUpperCase();letterOnly(this);'></input>"
        content = content + "</div>"
    }
    divAlternativas.innerHTML = content
}

function insertOutras(quantOut){

    var divOutras = document.getElementById("outras");
    divOutras.innerHTML = ""
    content = ""
    
    for (var i = 1; i <= quantOut; i++) {

        if (i < 10){
            pos = "0" + String(i);
        }
        else{
            pos = String(i);
        }

        content = content + "<div class = 'resposta'>"
        content = content + "<label for='out"+ (parseInt(pos) + 30) +"'>"+ (parseInt(pos) + 30) +"</label>"
        content = content + "<input type='text' id='out"+ (parseInt(pos) + 30) +"' name='out"+ (parseInt(pos) + 30) +"' placeholder='máx: 6' maxlength='6' oninput='numberOnly(this)'></input>"
        content = content + "</div>"
    }
    divOutras.innerHTML = content

}