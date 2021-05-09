ignore_words = ['o', 'su', 'ar', 'kuo', 'kaip', 'kur', 'apie', 'koks', 'ka', 'kuom', 'ko', 'kokiais', 'pasibaigia', 'daryti', 'buti', 'reikia', 'zinoti', 'rasyti',
 'yra', 'kokiu', 'kas', 'ir', 'per', 'kokia', 'esi', 'kuriuo', 'kokios', 'kokio', 'nors', 'galeti', 'rasti', 'rastas', 'galimybe', 'kuriuoje', 'ziureti', 'tureti', 'universitetas',
'turi', 'uz', 'labai', 'del', 'kuri', 'kokie', 'prie', 'kokias', 'koki', 'buvo', 'i', 'gauti', 'jei', 'jeigu', 'kelinta', 'kada', 'noreti', 'matyti', 'kokiomis', 'kiek', 'rodyti', 'informacija']

def lem(word):
        if(word == 'sveikas' or word == 'sveika' or word == 'laba' or word == 'labas' or word == 'hello' or word == 'hey' or word == 'sveiki'):
            return 'sveikinimas'
        elif(word == 'grupes' or word == 'grupei' or word == 'grupiu' or word == 'grupems'):
            return 'grupe'
        elif(word == 'varda' or word == 'vardu' or word == 'vardui' or word == 'vadinti'):
            return 'vardas'
        elif(word == 'vakaro' or word == 'vakarui' or word == 'vakara' or word == 'vakarai'):
            return 'vakaras'
        elif(word == 'gali' or word == 'galejo' or word == 'galejau' or word == 'galimai' or word == 'gali' or word == 'galima' or word == 'galiu' or word == 'galeciau' or word == 'negaliu'):
            return 'galeti'
        elif(word == 'noriu' or word == 'nori' or word == 'norejau' or word == 'norejo'):
            return 'noreti'
        elif(word == 'taves' or word == 'tave' or word == 'tau' or word == 'tavo'):
            return 'tu'
        elif(word == 'paskaitos' or word == 'paskaitai' or word == 'paskaitu' or word == 'paskaitoms' or word == 'paskaitomis' or word == 'paskaitai'):
            return 'paskaita'
        elif(word == 'naujena' or word == 'naujienos' or word == 'naujenos' or word == 'naujienai' or word == 'nauji' or word == 'naujos' or word == 'naujienoms' or word == 'naujienas' or word == 'naujo' or word == 'naujenas'):
            return 'naujas'
        elif(word == 'dienos' or word == 'dienai' or word == 'dienoms'):
            return 'diena'
        elif(word == 'geros' or word == 'gerai' or word == 'geriems' or word == 'geriausio'):
            return 'geras'
        elif(word == 'zinai' or word == 'zino' or word == 'zinojimas' or word == 'suzinoti'):
            return 'zinoti'
        elif(word == 'tvarkarasciai' or word == 'tvarkarascius' or word == 'tvarkarasti' or word == 'tvarkarasciui'):
            return 'tvarkarastis'
        elif(word == 'kalendoriu' or word == 'kalendoriui' or word == 'kalendoriai' or word == 'kalendoriuje'):
            return 'kalendorius'
        elif(word == 'aprasa' or word == 'aprasui' or word == 'aprasom' or word == 'aprasoms' or word == 'aprasas'):
            return 'aprasymas'
        elif(word == 'grupes' or word == 'grupei' or word == 'grupems' or word == 'grupemis'):
            return 'grupe'
        elif(word == 'studijoms' or word == 'studiju' or word == 'studijas'):
            return 'studijos'
        elif(word == 'kartai' or word == 'karta' or word == 'karto'):
            return 'kartas'
        elif(word == 'kiti' or word == 'kitiems' or word == 'kitie' or word == 'kito'):
            return 'kitas'
        elif(word == 'rodo' or word == 'rode' or word == 'parode' or word == 'rodomas' or word == 'rodomos' or word == 'parodyti' or word == 'atrodyti'):
            return 'rodyti'
        elif(word == 'randu' or word == 'radau' or word == 'narandu' or word == 'neradau' or word == 'randa' or word == 'neranda' or word == 'randasi' or word == 'rasiu' or word == 'nerandu' or word == 'surasti' or word == 'atrasti'):
            return 'rasti'
        elif(word == 'gerai' or word == 'geri' or word == 'geriems' or word == 'gero'):
            return 'geras'
        elif(word == 'pamatyti' or word == 'mato' or word == 'pamato' or word == 'mate' or word == 'pamate'):
            return 'matyti'
        elif(word == 'padejo' or word == 'padejai' or word == 'padeka'):
            return 'padeti'
        elif(word == 'turi' or word == 'turiu' or word == 'turejo' or word == 'turejau'):
            return 'tureti'
        elif(word == 'susijusias' or word == 'susijusios' or word == 'susije'):
            return 'sieti'
        elif(word == 'siu' or word == 'sio' or word == 'siuo' or word == 'sia' or word == 'siame'):
            return 'sis'
        elif(word == 'programos' or word == 'programu' or word == 'programai'):
            return 'programa'
        elif(word == 'universitete' or word == 'universiteto' or word == 'vgtu'):
            return 'universitetas'
        elif(word == 'savo' or word == 'mano' or word =='man'):
            return 'as'
        elif(word == 'isidarbinimas' or word == 'isidarbinimo' or word == 'idarbinimo'):
            return 'isidarbinti'
        elif(word == 'darbo' or word == 'darba' or word == 'darbu' or word == 'drabai' or word == 'darbus' or word == 'darbai' or word == 'darbui' or word == 'darbams'):
            return 'darbas'
        elif(word == 'auditorijos' or word == 'srl-i' or word == 'srk-ii' or word == 'srl-ii' or word == 'sra-i' or word == 'src' or word == 'sra-ii' or word == 'srk-i' or word == 'auditoriju'):
            return 'auditorija'
        elif(word == 'rezultatais' or word == 'rezultatai' or word == 'rezultatus' or word == 'rezultato'):
            return 'rezultatas'
        elif(word == 'fakultetai' or word == 'fakulteto' or word == 'fakulteta'):
            return 'fakultetas'
        elif(word == 'salygas' or word == 'salygos' or word == 'salygomis'):
            return 'salyga'
        elif(word == 'sarasa' or word == 'sarasai'):
            return 'sarasas'
        elif(word == 'stipendiju' or word == 'stipendijas' or word == 'stipendijos'):
            return 'stipendija'
        elif(word == 'studentu' or word == 'studentams' or word == 'studentui' or word == 'studento'):
            return 'studentas'
        elif(word == 'prasymu' or word == 'prasymus' or word == 'prasyma' or word == 'prasymo' or word == 'prasymai'):
            return 'prasymas'
        elif(word == 'pateiktu' or word == 'teikti' or word == 'pateikimo' or word == 'teikiamos' or word == 'teikia' or word == 'teikimas' or word == 'teikimos'):
            return 'pateikti'
        elif(word == 'kontaktus'):
            return 'kontaktai'
        elif(word == 'zemelapio' or word == 'zemelapiai' or word == 'zemelapius' or word == 'zemelapiu' or word == 'zemelapi'):
            return 'zemelapis'
        elif(word == 'gaunu' or word == 'gauni' or word == 'gavau' or word == 'gavimo' or word == 'gaunama' or word == 'gautas'):
            return 'gauti'
        elif(word == 'skelbimai' or word == 'skelbimui' or word == 'skelbimus'):
            return 'skelbimas'
        elif(word == 'ismokos' or word == 'ismokamos' or word == 'ismoketas' or word == 'ismokas' or word == 'ismoku' or word == 'ismokama'):
            return 'ismoka'
        elif(word == 'gatveje'):
            return 'gatve'
        elif(word == 'korpusa' or word == 'korpuso'):
            return 'korpusas'
        elif(word == 'rumus'):
            return 'rumai'
        elif(word == 'dekanato' or word == 'prodekanu' or word == 'dekanatu' or word == 'dekanui'):
            return 'dekanatas'
        elif(word == 'siulo' or word == 'pasiulimus' or word == 'pasiulymai' or word == 'pasiulymu' or word == 'siulymai' or word == 'siula' or word == 'siulymas'):
            return 'siulyti'
        elif(word == 'galimybes' or word == 'galimybiu'):
            return 'galimybe'
        elif(word == 'karjeros'):
            return 'karjera'
        elif(word == 'anketas'):
            return 'anketa'
        elif(word == 'ivertinimus' or word == 'ivertinimai' or word == 'vertinimo' or word == 'ivertinimu' or word == 'ivertinami' or word == 'vertinami' or word == 'ivertinimo'):
            return 'ivertinimas'
        elif(word == 'automobiliu'):
            return 'automobilis'
        elif(word == 'darbuotoju' or word == 'darbuotojais'):
            return 'darbuotojas'
        elif(word == 'grupioku' or word == 'grupiokais'):
            return 'grupiokas'
        elif(word == 'tarpinius' or word == 'tarpiniai'):
            return 'tarpinis'
        elif(word == 'egzaminu' or word == 'egzaminai' or word == 'egzaminus' or word == 'egzamino'):
            return 'egzaminas'
        elif(word == 'perziureti' or word == 'paziureti'):
            return 'ziureti'
        elif(word == 'destytoju' or word == 'desytojai'):
            return 'destytojas'
        elif(word == 'dalyku' or word == 'dalykus' or word == 'dalyko'):
            return 'dalykas'
        elif(word == 'adresu' or word == 'adresai'):
            return 'adresas'
        elif(word == 'sales'):
            return 'sale'
        elif(word == 'svetaineje'):
            return 'svetaine'
        elif(word == 'surasyti' or word == 'rasomi' or word == 'rasomos' or word == 'parasyti' or word == 'rasoma'):
            return 'rasyti'
        elif(word == 'mokslo' or word == 'mokslu'):
            return 'mokslas'
        elif(word == 'bendrabuciai'):
            return 'bendrabutis'
        elif(word == 'pasimatymo'):
            return 'pasimatymas'
        elif(word == 'dekui'):
            return 'aciu'
        elif(word == 'paklausi' or word == 'pasiklausti'):
            return 'klausti'
        elif(word == 'ziniu'):
            return 'zinios'
        elif(word == 'naudingas' or word == 'naudinga'):
            return 'nauda'
        elif(word == 'padeti' or word == 'pagalbos'):
            return 'pagalba'
        elif(word == 'vyksta' or word == 'ivyko'):
            return 'vykti'
        elif(word == 'paskutines'):
            return 'paskutinis'
        elif(word == 'metu' or word == 'metai'):
            return 'metas'
        elif(word == 'busenas'):
            return 'busena'
        elif(word == 'sporto'):
            return 'sportas'
        elif(word == 'sauletekyje' or word == 'sauletekio'):
            return 'sauletekis'
        elif(word == 'varslo'):
            return 'verslas'
        elif(word == 'gamtos'):
            return 'gamta'
        elif(word == 'architekturos'):
            return 'architektura'
        elif(word == 'kurybiniu'):
            return 'kuryba'
        elif(word == 'aviacijos'):
            return 'aviacija'
        elif(word == 'skyrimo' or word == 'skyriama'):
            return 'skirymas'
        elif(word == 'semestro' or word == 'semestre'):
            return 'semestras'
        elif(word == 'elektronikos'):
            return 'elektronika'
        elif(word == 'fundamentiniu' or word == 'fmf'):
            return 'fundamentinis'
        elif(word == 'inzinerijos'):
            return 'inzinerija'
        elif(word == 'ivairius'):
            return 'ivairus'
        elif(word == 'neiseina'):
            return 'neiseiti'
        elif(word == 'centrinius'):
            return 'centrinis'
        elif(word == 'bibliotekos'):
            return 'biblioteka'
        elif(word == 'nueiti'):
            return 'eiti'
        elif(word == 'palyginti'):
            return 'lyginti'
        elif(word == 'instituto'):
            return 'institutas'
        elif(word == 'laboratoriju'):
            return 'laboratorija'
        elif(word == 'industriju'):
            return 'industrija'
        elif(word == 'antano'):
            return 'antanas'
        elif(word == 'gustaicio'):
            return 'gustaitis'
        elif(word == 'vadybos'):
            return 'vadyba'
        elif(word == 'uzpildyti' or word == 'pildoma'):
            return 'pildyti'
        elif(word == 'mechnikos'):
            return 'mechnika'
        elif(word == 'katedros' or word == 'katedru'):
            return 'katedra'
        elif(word == 'pilna'):
            return 'pilnas'
        elif(word == 'statybos'):
            return 'statyba'
        elif(word == 'pavalgyti'):
            return 'valgyti'
        elif(word == 'dydzio'):
            return 'dydis'
        elif(word == 'treniruokliu'):
            return 'treniruoklis'
        elif(word == 'informacijos'):
            return 'informacija'
        elif(word == 'miesteli'):
            return 'miestas'
        elif(word == 'statusus'):
            return 'statusas'
        elif(word == 'menesio'):
            return 'menuo'
        elif(word == 'personalo'):
            return 'personalas'
        elif(word == 'anglu'):
            return 'angliskas'
        elif(word == 'kalbos'):
            return 'kalba'
        elif(word == 'gynimo' or word == 'gynimui'):
            return 'gynimas'
        elif(word == 'tvarkos'):
            return 'tvarka'
        elif(word == 'parengti' or word == 'rengiamas' or word == 'rengimo' or word == 'rengiami'):
            return 'rengti'
        elif(word == 'referata'):
            return 'referatas'
        elif(word == 'bendra' or word == 'bendri'):
            return 'bendras'
        elif(word == 'baigiamuju' or word == 'baigiamuosius' or word == 'baigiamaji' or word == 'baigiamojo' or word == 'baigiamajam'):
            return 'baigiamasis'
        elif(word == 'reikalavimai' or word == 'reikalavimus'):
            return 'reikalavimas'
        elif(word == 'rasto'):
            return 'rastas'
        elif(word == 'pristatymo' or word == 'pristatymu'):
            return 'pristatymas'
        elif(word == 'magistro'):
            return 'magistras'
        elif(word == 'prezentacijos' or word == 'prezentaciju'):
            return 'prezentacija'
        elif(word == 'praktikos'):
            return 'praktika'
        elif(word == 'pazeidimai'):
            return 'pazeidimas'
        elif(word == 'vieno' or word == 'pirmas' or word == 'pirma' or word == 'trecia' or word == 'trys' or word == 'vienas' or word == 'antra' or word == 'du' or word == 'ketvirta' or word == 'keturi' or word == 'penkta' or word == 'penki' or word == 'sesta' or word == 'sesi' or word == 'septinta' or word == 'septyni'):
            return 'skaicius'
        elif(word == 'puslapio'):
            return 'puslapis'
        elif(word == 'bakalauro'):
            return 'bakalauras'
        elif(word == 'pazymiai' or word == 'pazymiu'):
            return 'pazymys'
        elif(word == 'pasako'):
            return 'sakyti'
        elif(word == 'organizuojami' or word == 'organizuojama'):
            return 'organizuoti'
        elif(word == 'apeliuoja' or word == 'apeliacija' or word == 'apeliaciju' or word == 'apeliacijos'):
            return 'apeliuoti'
        elif(word == 'teigiamo'):
            return 'teigiamas'
        elif(word == 'galutiniai' or word == 'galutiniu' or word == 'galutinio'):
            return 'galutinis'
        elif(word == 'atspausdinti' or word == 'spausdinimo' or word == 'spausdinimas' or word == 'spausdinto'):
            return 'spausdinti'
        elif(word == 'atsiskaitymu' or word == 'atsiskaitymai' or word == 'atsiskaitymo' or word == 'atsiskaitoma'):
            return 'atsiskaitymas'
        elif(word == 'dokumenta' or word == 'dokumento'):
            return 'dokumentas'
        elif(word == 'skales'):
            return 'skale'
        elif(word == 'sablona' or word == 'sablonus' or word == 'sablonu' or word == 'sablonai'):
            return 'sablonas'
        elif(word == 'sesijos' or word == 'sesijas'):
            return 'sesija'
        elif(word == 'prasideda' or word == 'pradzia'):
            return 'prasideti'
        elif(word == 'laiko'):
            return 'laikas'
        elif(word == 'atlieka'):
            return 'atlikti'
        elif(word == 'blanka' or word == 'blankai'):
            return 'blankas'
        elif(word == 'sudaromas' or word == 'sudaro'):
            return 'daryti'
        elif(word == 'apibudinimus'):
            return 'apibudinimas'
        elif(word == 'valandos'):
            return 'valanda'
        elif(word == 'nesutinki'):
            return 'nesutikti'
        elif(word == 'islaikytas' or word == 'islaikymo' or word == 'neislaikymo' or word == 'neislaikytas'):
            return 'islaikyti'
        elif(word == 'pietu'):
            return 'pietus'
        elif(word == 'pertraukos' or word == 'pertrauku'):
            return 'pertrauka'
        elif(word == 'atrodo' or word == 'atrode'):
            return 'atrodyti'
        elif(word == 'tinkleli'):
            return 'tinklelis'
        elif(word == 'paslaugos' or word == 'paslaugas' or word == 'paslaugu' or word == 'paslaugomis'):
            return 'paslauga'
        elif(word == 'gebejimus'):
            return 'gebejimas'
        elif(word == 'priezasciu' or word == 'priezastys'):
            return 'priezastis'
        elif(word == 'kainuoja' or word == 'kaina' or word == 'kainos'):
            return 'kainuoti'
        elif(word == 'kursu'):
            return 'kursas'
        elif(word == 'laikus' or word == 'laika'):
            return 'laikas'
        elif(word == 'datos' or word == 'datu'):
            return 'data'
        elif(word == 'imonese' or word == 'imones'):
            return 'imone'
        elif(word == 'vietoje'):
            return 'vieta'
        elif(word == 'visa' or word == 'visus' or word == 'visu' or word == 'visi'):
            return 'visas'
        elif(word == 'nespalvotas' or word == 'spalvoto' or word == 'nespalvoto'):
            return 'spalvotas'
        elif(word == 'lapo'):
            return 'lapas'
        elif(word == 'atvejeis' or word == 'atvejai' or word == 'atveju' or word == 'atvejais'):
            return 'atvejis'
        elif(word == 'prasimu'):
            return 'prasimas'
        elif(word == 'israsomos' or word == 'israsas' or word == 'israsyta'):
            return 'israsyti'
        elif(word == 'atostogas'):
            return 'atostogos'
        elif(word == 'naudojama' or word == 'pasinaudoti'):
            return 'naudotis'
        elif(word == 'perlaikymai' or word == 'perlaikymu'):
            return 'perlaikymas'
        else:
            return word
