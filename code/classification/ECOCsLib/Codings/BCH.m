%##########################################################################
% BCH code for (ECOClib Sergio Escalera)
%##########################################################################
% input: number of rows
%        BCH code length
% output: ECOC - BCH codes of rows with maximum correction capability
%         t - correction capability
%##########################################################################

function [ECOC,t]=BCH(rows_number, code_len)

msg_len=ceil(log2(rows_number));
T=bchnumerr(code_len);
k=T(max(find(T(:,2)>=msg_len)),2);
t=T(max(find(T(:,2)>=msg_len)),3);

if isempty(k)
    error('Wrong BCH coding parameters');
end

msg=gf([flipud(dec2bin([0:rows_number-1])')'-'0', zeros(rows_number,k-msg_len)]);
code=bchenc(msg, code_len, k);

ECOC=2*double(code.x)-1;

for z=1:code_len
    if sum(ECOC(:,z)==1)==0
        ECOC(1,z)=1;
    elseif sum(ECOC(:,z)==-1)==0
        ECOC(1,z)=-1;
    end
end
