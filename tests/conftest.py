import pytest


@pytest.fixture
def example_model_input_content():
    return [
        "臺灣高雄地方法院民事判決110年度雄簡字第844號原告王梅珠訴訟代理人鄭心愉被告陶聖翔訴訟代理人李恆瑋林建宏上列當事人間損害賠償事件，原告提起刑事附帶民事訴訟，經本院刑事庭移送前來（109年度交附民字第75號），本院於民國110年11月8日辯論終結，判決如下：主文被告應給付原告新臺幣98,532元及自民國109年10月27日起至清償日止按週年利率百分之5計算之利息。訴訟費用由被告負擔百分之15，餘由原告負擔。本判決第1項得假執行。但被告以新臺幣98,532元為原告預供擔保後得免為假執行。事實及理由一、原告主張：被告於民國108年10月14日11時50分許，駕駛車牌號碼000-0000號自用小客車，沿高雄市新興區中山一路南向北行駛，並在中山一路與中山橫路交會之路口停等紅燈。疏未注意路口東西向行人穿越道之動態，未待紅燈轉為綠燈即放煞車，並在燈號轉為綠燈後旋即起步。適有原告騎乘自行車沿中山橫路西向東行經上開路口，因此發生碰撞，致原告受有頭部外傷合併頭皮撕裂傷及腦震盪之傷害，支出醫療費用新臺幣(下同)1,680元、看護費66,000元、交通費1,080元、鑑定費5,000元，並受有營業損失372,726元及非財產上慰撫金損害200,000元，均應由被告賠償之。爰依侵權行為法律關係提起本件訴訟。並聲明：被告應給付原告646,486元(腳踏車部分不請求)及自起訴狀繕本送達翌日起至清償日止按週年利率百分之5計算之利息。二、被告則以：原告未在綠燈時段通過路口，被告不及注意，本件事故發生原告亦與有過失。原告請求之醫療費用、交通費均不爭執，同意賠付。看護費及營業損失部分期間1個月不爭執，超過1個月部分被告不同意，且原告未證明營業金額；鑑定費是原告自行送鑑定，不同意賠付，慰撫金請求過高等詞置辯。並聲明：原告之訴駁回。三、兩造不爭執事項㈠被告在108年10月14日11時50分許駕駛車輛，沿高雄市新興區中山一路南向北行駛，並在中山一路與中山橫路交會之路口停等紅燈，但未待紅燈轉為綠燈即放煞車，並在燈號轉為綠燈後旋即起步。適有原告騎乘自行車沿中山橫路西向東行經上開路口，未及在該行向燈號轉為紅燈前通過，雙方閃煞不及發生碰撞，原告因而人車倒地，受有頭部外傷合併頭皮撕裂傷及腦震盪之傷害。㈡原告因上開事故對被告提起告訴，經本院109年度交易字第67號認被告犯過失傷害罪，處拘役30日，得易科罰金。被告不服提起上訴，經臺灣高等法院高雄分院110年交上易字第69號刑事判決駁回被告上訴，已告確定(下稱系爭刑案)。㈢原告尚未領強制險。四、兩造爭執事項㈠原告是否與有過失？兩造過失比例為何？㈡原告得請求之金額若干為適當?五、得心證之理由㈠兩造就本件事故發生均有過失，被告肇事責任為百分之70，原告為百分之30。⒈按汽車行駛時，駕駛人應注意車前狀況，並隨時採取必要之安全措施，道路交通安全規則第94條第3項定有明文。系爭刑案勘驗被告之行車紀錄器畫面顯示：（畫面時間00：00：05至00：00：06）被告之車輛持續緩慢向前滑行，路口為綠燈，原告正由西往東騎乘自行車通過外環行人專用道之斑馬線；（畫面時間00：00：07至00：00：08）被告車輛超過左側車輛後，仍持續緩慢向前滑行，與原告距離越來越近，惟並無煞車之跡象（見系爭刑案交上易卷第57頁至第58頁、第63頁至第67頁）。足見雙方發生碰撞前，被告之車輛已向前行駛超越同向之左側車輛，視線不受該車阻擋，佐以被告自陳其車速當時僅有時速5公里等語（見系爭刑案警卷第35頁），堪認被告實有足夠之反應時間以肉眼發現車輛前方之告訴人，並無不能注意之情事甚明。駕駛即使在燈號甫轉換為綠燈之際，仍應注意車前狀況，非謂取得路權後，自此即可免除交通安全之注意義務，此乃至明之理。而以一般駕駛人之注意義務程度，均能避免與原告發生碰撞。可徵被告對於本案車禍事故之發生，顯有疏未注意車前狀況之過失。⒉次按慢車行駛時，駕駛人應注意車前狀況，並隨時採取必要之安全措施，道路交通安全規則第124條第5項定有明文。本件原告越過停止線至上開路口，相對位置約在中山一路北向南，從外側數來第2、3道車道時，其行向已轉為紅燈乙情，並於通過路口時以右手抓著右側帽緣之方式騎乘自行車，遮蔽右側之視野有行車紀錄器影像截圖1張附卷可考（見系爭刑案交上易卷第57頁、第63頁）。是原告在無從確保自己得在燈號轉為紅燈前通過該路口之情形下，選擇貿然進入路口，自應加速通過該路口，並小心留意車前狀況，隨時採取必要之安全措施，以避免與他向來車發生車禍。惟原告就周遭、車前狀況疏為注意，更自行遮蔽右側之視野，以致其全然未發覺右前側之被告車輛已向前行駛，其對路權消逝後又未負起相當之注意義務，顯有過失。⒊兩造就本件事故發生均有過失，已如前述。審酌兩造注意義務、過失情節及避免事故發生之可能性等，認被告應負較大過失責任為百分之70，原告為百分之30。㈡原告得請求之金額若干為適當?⒈按因故意或過失，不法侵害他人之權利者，負損害賠償責任，民法第184條第1項前段定有明文。次按不法侵害他人之身體或健康者，對於被害人因此喪失或減少勞動能力或增加生活上之需要時，應負損害賠償責任；不法侵害他人之身體、健康、名譽、自由、信用、隱私、貞操，或不法侵害其他人格法益而情節重大者，被害人雖非財產上之損害，亦得請求賠償相當之金額，同法第193條第1項、第195條第1項規定甚明。被告就本件事故發生為有過失，並造成原告受有傷害，自應負損害賠償責任。茲就原告所得請求之金額，分述如下：①醫療費用：原告主張因本件事故支出醫藥費用1,680元，業據其提出診斷證明書、醫療費用收據等件為證，並為被告不爭執，表示同意賠付(見本院卷第158頁)故原告請求被告賠償醫療費用1,680元，應屬有據。②看護費用：原告主張需看護2個月，看護費66,000元，僅受訴外人葉秀桃看護，並提出診斷證明書、看護費用收據為憑(見附民卷第9頁、本院卷第49頁)。由上開診斷證明書可見醫師囑言記載原告108年10月14日住院至108年10月22日出院，住院中及出院後需休養及專人照顧1個月等語。是以原告所受之傷害自108年10月14日起1個月又5日有受看護之必要。審酌原告提出之上開看護費收據108年10月14日起至108年10月21日為每日2,000元；108年10月23日起為每日1,300元，據此計算原告得請求被告賠償之看護費應為53,000元(108年10月14日起至108年10月21日每日2,000元共14,000元+108年10月23日起至108年11月21日每日1,300元共39,000元，合計53,000元)。逾此部分無理由。③交通費：原告主張因本件事故支出交通費1,080元，雖未提出單據佐證，然為被告不爭執並同意賠付(見本院卷第159頁)，故原告此部分請求為有理由。④鑑定費：原告主張本件訴訟前支出鑑定費用共5,000元，並提出收據為證(見附民卷第55、56頁)。被告雖抗辯係原告自行鑑定，不應由被告負擔。然原告此部分支出應屬證明事故肇責並請求賠償所需之支出，故此部分請求，應予准許。⑤營業損失：原告主張不能工作2個月，為被告所爭執。審酌上開診斷證明書記載之休養期間為住院及出院後1個月共35日，是逾此部分之時間難認有不能工作之情形。另原告雖主張為肯倫大亨精品服飾行負責人，因此2個月無營業收入云云，惟上開商行負責人為王保全；且自國稅局亦無休業或歇業資料，又原告經通知應補正其為經營者，及有休業之相關事證均未提出，經查詢原告108年度名下亦無所得資料。再由原告所提之銷貨資料可見每月銷售額本不一定，原告也沒有證明上開期間有因此短少收入，故難認原告受有營業損失，原告此部分請求，礙難准許。⑦慰撫金：原告因事故受有上開身體傷害，請求慰撫金，自屬有據。審酌原告所受之傷害及兩造名下收入等一切情狀，原告請求慰撫金80,000元為適當，逾此範圍之請求過高，不應准許。⒉末按損害之發生或擴大，被害人與有過失者，法院得減輕賠償金額或免除之，為民法第217條第1項所明定。兩造就本件事故發生均有過失，已如前述，是依上開規定得減輕被告賠償金額之百分之30，經按前開過失比例減輕後，被告應賠償原告之金額應為98,532元（醫療費用1,680元、看護費53,000元、交通費1,080元、鑑定費5,000元、慰撫金80,000元，合計140,760×百分之70=98,532元)。逾此範圍請求無理由。六、從而，原告依侵權行為法律關係請求被告給付原告98,532元及自起訴狀繕本送達翌日即109年10月27日起至清償日止按週年利率百分之5計算之利息為有理由，應予准許。逾此範圍之請求為無理由，應予駁回。七、本件係就民事訴訟法第427條訴訟適用簡易程序所為被告敗訴之判決，爰依同法第389條第1項第3款之規定，就原告勝訴部分職權宣告假執行。另依職權諭知免為假執行之擔保金額。八、至兩造其餘之攻擊防禦方法及未經援用之證據，經本院斟酌後，認為均不足以影響本判決之結果，自無逐一詳予論駁之必要，併此敘明。九、據上論結，原告之訴一部有理由一部無理由，依民事訴訟法第436條第2項、第385條第1項前段、第79條、第389條第1項第3款、第392條第2項規定，判決如主文。中華民國110年11月22日高雄簡易庭法官楊詠惠以上正本係照原本作成。如不服本判決，應於送達後20日內，向本院提出上訴狀並表明上訴理由，如於本判決宣示後送達前提起上訴者，應於判決送達後20日內補提上訴理由書（須附繕本）。中華民國110年11月22日書記官蔡佩珊"
    ]